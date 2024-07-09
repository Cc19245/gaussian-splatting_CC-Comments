/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>   // CUDA的CUB库
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>   // GLM (OpenGL Mathematics)库

#include <cooperative_groups.h>   // CUDA 9引入的Cooperative Groups库
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// 寻找给定无符号整数 n 的最高有效位（Most Significant Bit, MSB）的下一个最高位
// Helper function to find the next-highest bit of the MSB
// on the CPU.
// 查找最高有效位(most significant bit）, 输入变量n表示tile编号最大值x、y的乘积
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;  //4*4=16
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;      // 缩小2倍
		if (n >> msb)   // 右移16位, 相当于除以2的16次方
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)       // 如果n的最高位大于0, 则msb+1
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,   // 所有点的个数
	const float* orig_points,   // 原始点的坐标（world系）
	const float* viewmatrix,    // w2c, 原始点的world系到cam系的变换矩阵
	const float* projmatrix,    // 投影矩阵
	bool* present)   // 结果保存到present中, 表示某个点是否在视锥中
{
	// 返回当前thread的值
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

/**
 * @brief 计算2d高斯椭圆中心点points_xy在2d像素平面上占据的tile的tileID, 
 *  并将tileID|depth组合成64位的key值, value值为高斯球的编号
 * 
 */
// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,   // 每个3D高斯中心点投影到图像上之后的2D坐标
	const float* depths,       // 每个3D高斯中心点从world系转移到相机系后的Z轴深度
	const uint32_t* offsets,   // 每个投影后的2D高斯覆盖的tiles数量的前缀和累积数组
	uint64_t* gaussian_keys_unsorted,    // 未排序的key（tileID|depth）
	uint32_t* gaussian_values_unsorted,  // 未排序的value, 即key对应的高斯球的idx
	int* radii,  // 每个3D高斯投影到图像上的2D高斯的最大半径（椭圆长轴）
	dim3 grid)   // dim3, 表示长宽方向要多少个tile才能覆盖整张图像
{  
	auto idx = cg::this_grid().thread_rank();  // 线程索引, 该线程处理第idx个Gaussian
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		//; 第idx个高斯球前面已经占据的tiles总数, 用于在gaussian_keys_unsorted和
		//; gaussian_values_unsorted中存储本次生成的key/value的索引起点
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;  // 占据网格的左上角位置、右下角位置

		// 因为要给Gaussian覆盖的每个tile生成一个(key, value)对, 所以先获取它占了哪些tile
		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

		// 对于边界矩形重叠的每个瓦片, 具有一个键/值对。
		// 键是 | 瓦片ID | 深度 |, 
		// 值是高斯的ID, 按照这个键对值进行排序, 将得到一个高斯ID列表, 
		// 这样它们首先按瓦片排序, 然后按深度排序
		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;  // tile的ID
				key <<= 32;  // 放在高位, 左移32位, 把后面32位空出来
				key |= *((uint32_t*)&depths[idx]);  // 低位是深度
				gaussian_keys_unsorted[off] = key;    // key是前面组合出来的key
				gaussian_values_unsorted[off] = idx;  // value是3D高斯的idx
				off++;  // 当前高斯占据的tile总数增加一个, 所以索引++
			}
		}
	}
}

// 识别每个瓦片（tile）在排序后的高斯ID列表中的范围
// 目的是确定哪些高斯ID属于哪个瓦片, 并记录每个瓦片的开始和结束位置
// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(
	int L,   // 排序列表中的元素个数
	uint64_t* point_list_keys,   // 排过序的keys
	uint2* ranges)   // ranges[tile_id].x和y表示第tile_id个tile在排过序的列表中的起始和终止索引
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;  // 当前tile ID 
	if (idx == 0) 
		ranges[currtile].x = 0;  // 边界条件：tile 0的起始位置
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		// 上一个元素和我处于不同的tile, 那我是上一个tile的终止位置和我所在tile的起始位置
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;  // 边界条件：最后一个tile的终止位置
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

// 在给定的内存块中初始化 GeometryState 结构
// chunk（一个指向内存块的指针引用）, P（元素的数量）
// 使用 obtain 函数为 GeometryState 的不同成员分配空间, 并返回一个初始化的 GeometryState 实例
CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	// 计算数组的前缀和,InclusiveSum表示包括自身, ExclusiveSum表示不包括自身
    // 参数：第三个in, 第四个out, 最后一个num。当第一个参数为NULL时, 所需的分配大小被写入第二个参数, 并且不执行任何工作
    // https://github.com/dmlc/cub/blob/master/cub/device/device_scan.cuh
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

// 初始化 BinningState 实例, 分配所需的内存, 并执行排序操作
CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	// 在 GPU 上进行基数排序, 将 point_list_keys_unsorted 作为键, point_list_unsorted 作为值进行排序, 排序结果存储在 point_list_keys 和 point_list 中
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, int D, int M,   // 点的个数, 当前激活的最大球谐次数, 所有球谐基函数的个数
	const float* background,     // [0, 0, 0], 背景颜色
	const int width, int height, // 要渲染的图像宽高
	const float* means3D,   // (P, 3), 高斯中心点位置
	const float* shs,       // (P, 16, 3), 球谐系数
	const float* colors_precomp,   // None
	const float* opacities,   // (P, 1), 高斯中心点的不透明度
	const float* scales,      // (P, 3), 尺度
	const float scale_modifier,  // 尺度缩放因子, 和场景大小有关
	const float* rotations,      // (P, 4), 旋转
	const float* cov3D_precomp,  // None
	const float* viewmatrix,     // (4, 4), W2C
	const float* projmatrix,     // (4, 4), 整个投影矩阵, 包括W2C和视角变换矩阵
	const float* cam_pos,        // (3, ), 相机中心位置
	const float tan_fovx, float tan_fovy,   // 相机FoV
	const bool prefiltered,      // False
	float* out_color,    // (3, H, W), 渲染的图像
	int* radii,   // (P, ), 每个高斯球投影到图像上的最大半径
	bool debug)
{
	const float focal_y = height / (2.0f * tan_fovy);  // 垂直方向的焦距 focal_y
	const float focal_x = width / (2.0f * tan_fovx);   // 水平方向的焦距 focal_x

	// Step 1. 为每个高斯球 GeometryState 和输出图像像素 ImageState 的数据结构分配内存空间
	// 模版函数required调用fromChunk函数来获取内存, 返回结束地址, 也即所需存储大小
	size_t chunk_size = required<GeometryState>(P);  
	// 调用rasterize_points.cu文件中的resizeFunctional函数里面嵌套的匿名函数lambda来调整显存块大小, 并返回首地址
	char* chunkptr = geometryBuffer(chunk_size);  
	// 在给定的内存块中初始化 GeometryState 结构体, 为不同成员分配空间, 并返回一个初始化的实例
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;  // 指向radii数据的指针
	}
	
	// dim3是CUDA定义的含义x,y,z三个成员的三维unsigned int向量类, tile_grid就是x和y方向上tile的个数
	// 定义了一个三维网格（dim3 是 CUDA 中定义三维网格维度的数据类型）, 确定了在水平和垂直方向上需要多少个块来覆盖整个渲染区域
	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	// 确定了每个块在 X（水平）和 Y（垂直）方向上的线程数
	dim3 block(BLOCK_X, BLOCK_Y, 1);  // 块中线程数量, 16*16

	// Dynamically resize image-based auxiliary buffers during training
	// 计算存储所有2D pixel的各个参数所需要的空间大小
	size_t img_chunk_size = required<ImageState>(width * height);
	// 给所有2D pixel的各个参数分配存储空间, 并返回存储空间的指针
	char* img_chunkptr = imageBuffer(img_chunk_size);
	// 在给定的内存块中初始化 ImageState 结构体, 为不同成员分配空间, 并返回一个初始化的实例
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// Step 2. 预处理：计算3D高斯中心点的深度、投影的像素位置、投影到图像上的像素颜色, 2D高斯的协方差矩阵和最大半径
	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	// 预处理, 主要涉及把3D的Gaussian投影到2D
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,  // 3D gaussian的个数, 当前激活的球谐函数的次数, 所有球谐系数的个数 
		means3D,  // (P, 3), 每个3D gaussian的XYZ均值
		(glm::vec3*)scales,  // (P, 3), 每个3D gaussian的XYZ尺度
		scale_modifier,   // 尺度缩放系数, 1.0
		(glm::vec4*)rotations,  // (P, 4), 每个3D gaussian的旋转四元组
		opacities,    // 每个3D gaussian的不透明度
		shs,   // (P, 16, 3), 每个3D gaussian的球谐系数, 用于表示颜色
		geomState.clamped,  // (P, 3), 存储每个3D gaussian的R、G、B是否小于0
		cov3D_precomp,   // 提前计算好的每个3D gaussian的协方差矩阵, []
		colors_precomp,  // 提前计算好的每个3D gaussian的颜色, []
		viewmatrix,   // (4, 4), 相机外参矩阵, world to camera
		projmatrix,   // (4, 4), 整个投影矩阵, 包括W2C和视角变换矩阵
		(glm::vec3*)cam_pos,  // (3,), 相机的中心点XYZ坐标
		width, height,        // 图像的宽和高
		focal_x, focal_y,     // 水平、垂直方向的焦距
		tan_fovx, tan_fovy,   // 水平、垂直视场角一半的正切值
		radii,    // (P,), 存储每个2D gaussian在图像上的半径
		geomState.means2D,   // (P, 2), 存储每个2D gaussian的均值
		geomState.depths,    // (P,), 存储每个2D gaussian的深度
		geomState.cov3D,     // (P, 6), 存储每个3D gaussian的协方差矩阵
		geomState.rgb,       // (P, 3), 存储每个2D pixel的颜色
		geomState.conic_opacity,  // (P, 4), 存储每个2D gaussian的协方差矩阵的逆矩阵以及它的不透明度
		tile_grid,  // 在水平和垂直方向上需要多少个块来覆盖整个渲染区域
		geomState.tiles_touched,  // (P,), 存储每个2D gaussian覆盖了多少个tile
		prefiltered    // False, 是否预先过滤掉了中心点(均值XYZ)不在视锥（frustum）内的3D gaussian
	), debug)

	// Step 2.使用InclusiveSums计算所有高斯覆盖的tile的总数, 并用于申请BinningState的显存
	// ---开始--- 通过视图变换 W 计算出像素与所有重叠高斯的距离, 即这些高斯的深度, 形成一个有序的高斯列表
	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	// 这步是为duplicateWithKeys做准备, 计算出每个Gaussian对应的keys和values在数组中存储的起始位置）
	// 同步运行InclusiveSum, 获取tiles_touched数组的前缀和, 存到point_offsets中
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(
		geomState.scanning_space, geomState.scan_size, 
		geomState.tiles_touched, geomState.point_offsets, P), debug)

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;  // 所有的2D gaussian总共覆盖了多少个tile
	// 将 geomState.point_offsets 数组中最后一个元素的值复制到主机内存中的变量 num_rendered
	// 把指针（point_offsets + P - 1）, 也就是point_offsets数组的最后一个元素的值, 赋给num_rendered, 也就是总共覆盖的tiles数量
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	// 计算所需的BinningState的数量, 即每个高斯球覆盖的tile都有对应的装箱状态BinningState数据
	size_t binning_chunk_size = required<BinningState>(num_rendered);
	// 调整显存块大小, 并返回首地址
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
    // 用显存块首地址作为参数, 调用fromChunk函数来申请显存
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);
	
	// Step 3. 生成tile和3D高斯深度组成的key / 3D高斯索引组成的value  的键值对
	// 将每个3D gaussian的对应的tile index和深度存到point_list_keys_unsorted中
    // 将每个3D gaussian的对应的index（第几个3D gaussian）存到point_list_unsorted中
	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (  // 根据 tile, 复制 Gaussian
		P,
		geomState.means2D,   // 每个3D高斯的中心点投影到图像上的像素坐标
		geomState.depths,    // 每个3D高斯中心点从world系转移到相机系后的Z轴深度
		geomState.point_offsets,    // 这里用到上步InclusiveSum得到的累计高斯球touch的tiles数
		binningState.point_list_keys_unsorted,  // 存储key（tileID|depth）
		binningState.point_list_unsorted,  // 存储对应的高斯球idx
		radii,   // 像素平面上高斯圆的半径, 最长轴的3倍
		tile_grid)  // 全图中tile的数量
	CHECK_CUDA(, debug)    // 同步, 并检查错误

	// 查找tile_grid.x * tile_grid.y的最高位
	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Step 4. 依据key进行排序, 这样根据tile的ID和深度值从小到大排序
    // https://nvlabs.github.io/cub/structcub_1_1_device_radix_sort.html
	// 对一个键值对列表进行排序。这里的键值对由 binningState.point_list_keys_unsorted 和 binningState.point_list_unsorted 组成
    // 排序后的结果存储在 binningState.point_list_keys 和 binningState.point_list 中
    // binningState.list_sorting_space 和 binningState.sorting_size 指定了排序操作所需的临时存储空间和其大小
    // num_rendered 是要排序的元素总数。0, 32 + bit 指定了排序的最低位和最高位, 这里用于确保排序考虑到了足够的位数, 以便正确处理所有的键值对
	// Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs( // 对复制后的所有 Gaussians 进行排序, 排序的结果可供平行化渲染使用
		binningState.list_sorting_space,  // 辅助空间
		binningState.sorting_size,   // 辅助空间大小
		// d_keys_in, d_keys_out
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		// d_values_in, d_values_out
		binningState.point_list_unsorted, binningState.point_list,
		// num_rendered是要排序的元素总数(总共覆盖的tiles数量), 开始bit位, 结束比特位
		num_rendered, 0, 32 + bit), debug)

	// 将 imgState.ranges 数组中的所有元素设置为 0, 长度为tile_grid.x * tile_grid.y * sizeof(uint2)
	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// Step 5. 计算每个瓦片（tile）在排序后的高斯ID列表中的范围
    // 目的是确定哪些高斯ID属于哪个瓦片, 并记录每个瓦片的开始和结束位置
	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		// 根据有序的Gaussian列表, 判断每个 tile 需要跟哪一个 range 内的 Gaussians 进行计算
		// 计算每个tile对应排序过的数组中的哪一部分
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)
	// ---结束--- 通过视图变换 W 计算出像素与所有重叠高斯的距离, 即这些高斯的深度, 形成一个有序的高斯列表

	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	// Step 6. 调用核心渲染函数, 具体实现在 forward.cu/renderCUDA
	CHECK_CUDA(FORWARD::render(
		tile_grid,   // 在水平和垂直方向上需要多少个block来覆盖整个渲染区域
		block,  // 每个block在 X（水平）和 Y（垂直）方向上的线程数
		imgState.ranges,  // 每个瓦片（tile）在排序后的高斯ID列表中的范围
		binningState.point_list,  // 排序后的3D gaussian的id列表
		width, height,  // 图像的宽和高
		geomState.means2D,  // (P, 2), 每个2D gaussian在图像上的中心点位置
		feature_ptr,   // (P, 3), 每个3D gaussian对应的RGB颜色
		geomState.conic_opacity,  // (P, 4), 每个2D gaussian的协方差矩阵的逆矩阵以及它的不透明度
		imgState.accum_alpha,  // [out], 渲染过程后每个像素的最终透明度或透射率值
		imgState.n_contrib,  // [out], 每个pixel的最后一个贡献的2D gaussian是谁
		background,  // 背景颜色
		out_color),  // [out], 输出图像
		debug)  

	return num_rendered;   // 所有高斯覆盖的tile的总数, 可以反应是否有高斯投影到图像上
}

// 产生对应于前向渲染过程所需的优化梯度
// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
	// 3D高斯个数, 当前激活的球谐函数次数, 所有球谐函数的最高次数, 所有3D高斯覆盖的tile的个数
	const int P, int D, int M, int R,  
	const float* background,  // 背景颜色
	const int width, int height,   // 图像宽高
	const float* means3D,   
	const float* shs,
	const float* colors_precomp,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* campos,
	const float tan_fovx, float tan_fovy,
	const int* radii,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	// 上面的参数都是前向传播的时候进行计算的中间变量
	const float* dL_dpix, // 上游梯度, loss对每个像素颜色的导数, 维度(3, H, W)
	float* dL_dmean2D,    // loss对Gaussian二维中心坐标的导数
	float* dL_dconic,     // loss对椭圆二次型矩阵的导数
	float* dL_dopacity,   // loss对不透明度的导数
	float* dL_dcolor,     // loss对Gaussian颜色的导数（颜色是从相机中心看向Gaussian的颜色）
	float* dL_dmean3D,    // loss对Gaussian三维中心坐标的导数
	float* dL_dcov3D,     // loss对Gaussian三维协方差矩阵的导数
	float* dL_dsh,        // loss对Gaussian的球谐系数的导数
	float* dL_dscale,     // loss对Gaussian的缩放参数的导数
	float* dL_drot,       // loss对Gaussian旋转四元数的导数
	bool debug)
{
	// 下面这些缓冲区都是在前向传播的时候存下来的, 现在拿出来用
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);
	
	// 根据每像素损失梯度计算损失梯度, 关于2D均值位置、圆锥矩阵、
	// 高斯的不透明度和RGB。如果我们获得了预计算的颜色而不是球谐系数, 就使用它们。
	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		color_ptr,
		imgState.accum_alpha,
		imgState.n_contrib,
		dL_dpix,
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor), debug)

	// 处理预处理的剩余部分
	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	// 因为是反向传播, 所以preprocess放在后面了
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dconic,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		// 以下是输出变量
		dL_dcov3D,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot), debug)
}