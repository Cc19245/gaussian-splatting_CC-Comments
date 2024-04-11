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

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// 从每个3D gaussian对应的球谐系数中计算对应的颜色
// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx,   // 该线程负责第几个Gaussian
	int deg,   // 当前激活的球谐的度数  
	int max_coeffs,   // 使用的球谐基函数的总数
	const glm::vec3* means,   // Gaussian中心位置，world系下的
	glm::vec3 campos,   // 当前相机位置
	const float* shs,   // (P, 16, 3), 所有的球谐系数
	bool* clamped)   // 表示每个值是否被截断了（RGB只能为正数），这个在反向传播的时候用
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	//; 这里的索引挺有意思，注意(glm::vec3*)shs就强制把flaot* shs的内存转化成(glm::vec3*)类型，也就是
	//; 3个float一组表示一个vec3（RGB三个颜色的球谐系数值），然后每个3D高斯有max_coeffs个球谐基函数，
	//; 所以当前第idx个3D高斯的球谐系数就在内存起点的基础上偏移idx * max_coeffs个位置
	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];  // 直流分量

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(
	const float3& mean,   // Gaussian中心坐标
	float focal_x, float focal_y,   // x,y方向焦距
	float tan_fovx, float tan_fovy, // x,y方向视角
	const float* cov3D,   // 已经算出来的三维协方差矩阵
	const float* viewmatrix)   // W2C矩阵
{  
	// 这里描述了《EWA Splatting》（Zwicker等人，2002年）中方程29和31所概述的步骤。
	// 此外，还考虑了视口的宽高比/缩放比例。
	// 使用转置操作来适应行主序和列主序的约定。
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.

	// W2C矩阵乘Gaussian中心坐标得其在相机坐标系下的坐标
	float3 t = transformPoint4x3(mean, viewmatrix);  

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;  // Gaussian中心在像平面上的x坐标
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	// 透视变换是非线性的，因为一个点的屏幕空间坐标与其深度（Z值）成非线性关系。
	// 雅可比矩阵 J 提供了一个在特定点附近的线性近似，这使得计算变得简单且高效
	//! 疑问：这里的J为什么会有focal_x, focal_y这一项？
	//; 解答：因为这里最终计算的是投影到图像上的2D高斯的协方差矩阵，从而方便后面判断它覆盖了哪些tile
	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);  

	glm::mat3 W = glm::mat3(   // W2C矩阵
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(  // 3D协方差矩阵，是对称阵
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	// 这里的转置跟存储的时候行优先和列优先有关，没太看懂
	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	// 应用低通滤波器：每个高斯函数应至少为一个像素宽/高。丢弃第三行和列
	// 也没太看懂这里+0.3是为什么
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// 根据当前3D gaussian的尺度和旋转参数计算其对应的协方差矩阵
// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);  // 初始化了一个3x3的单位阵
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}


/**
 * @brief 渲染前的预处理函数，实现如下功能：
 *   1.计算3D高斯中心在相机坐标系下的深度、投影到图像上的像素点位置
 *   2.计算2D高斯在像素平面上的协方差矩阵，并进行特征值分解得到长短轴（最大半径）
 *   3.根据2D高斯的最大半径，计算它覆盖了哪些tile，并最终保存它覆盖了几个tile
 *   4.利用球谐系数计算3D高斯中心点投影到图像上的像素的颜色
 */
// Perform initial steps for each Gaussian prior to rasterization.
// 这里的模板参数C就是图像的channel通道个数
template<int C>
// __global__表示CPU调用，但是在GPU上执行
__global__ void preprocessCUDA(
	int P, int D, int M,  // 3D gaussian的个数, 当前激活的球谐函数的次数, 所有球谐系数的个数 
	const float* orig_points,  // (P, 3), 每个3D gaussian的XYZ均值
	const glm::vec3* scales,   // (P, 3), 每个3D gaussian的XYZ尺度
	const float scale_modifier,  // 尺度缩放系数, 1.0
	const glm::vec4* rotations,  // (P, 4), 每个3D gaussian的旋转四元组
	const float* opacities,  // (P,), 每个3D gaussian的不透明度
	const float* shs,  // (P, 16, 3), 每个3D gaussian的球谐系数, 用于表示颜色
	bool* clamped,  // (P, 3), 存储每个3D gaussian的R、G、B是否小于0
	const float* cov3D_precomp,    // 提前计算好的每个3D gaussian的协方差矩阵, []
	const float* colors_precomp,   // 提前计算好的每个3D gaussian的颜色, []
	const float* viewmatrix,   // (4, 4), 相机外参矩阵, world to camera
	const float* projmatrix,   // (4, 4), 整个投影矩阵，包括W2C和视角变换矩阵
	const glm::vec3* cam_pos,  // (3,), 相机的中心点XYZ坐标
	const int W, int H,   // 图像的宽和高
	const float tan_fovx, float tan_fovy,  // 水平、垂直视场角一半的正切值
	const float focal_x, float focal_y,    // 水平、垂直方向的焦距
	int* radii,   // (P,), 存储每个2D gaussian在图像上的半径
	float2* points_xy_image,   // (P, 2), 存储每个2D gaussian的均值
	float* depths,   // // (P,), 存储每个2D Gaussian中心的深度，即其在相机坐标系的z轴的坐标
	float* cov3Ds,   // (P, 6), 存储每个3D gaussian的协方差矩阵
	float* rgb,      // (P, 3), 根据球谐算出的每个2D pixel的RGB颜色值
	float4* conic_opacity,    // (P, 4), 存储每个2D gaussian的协方差矩阵的逆矩阵以及它的不透明度
	const dim3 grid,          // dim3, 在水平和垂直方向上需要多少个tile来覆盖整个渲染区域
	uint32_t* tiles_touched,  // (P,), 存储每个2D gaussian覆盖了多少个tile
	bool prefiltered)  // False, 是否预先过滤掉了中心点(均值XYZ)不在视锥（frustum）内的3D gaussian
{
	// 每个线程处理一个3D gaussian, index超过3D gaussian总数的线程直接返回, 防止数组越界访问
	auto idx = cg::this_grid().thread_rank();  // 该函数预处理第idx个Gaussian
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Step 1. 判断当前高斯是否在相机视野内，如果不再则不用处理
	// 给定指定的相机姿势，此步骤确定哪些3D高斯位于相机的视锥体之外。这样做可以确保在后续计算中不涉及给定视图之外的3D高斯，从而节省计算资源。
	// 判断当前处理的3D gaussian的中心点(均值XYZ)是否在视锥（frustum）内, 如果不在则直接返回
	// Perform near culling, quit if outside.
	float3 p_view;  //; 用于存储将 p_orig 通过位姿矩阵 viewmatrix 转换到相机坐标系下的坐标
	//; 注意实际只判断了点是否在相机前方
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting  以下代码将3D高斯（椭球）被投影到2D图像空间（椭圆），存储必要的变量供后续渲染使用
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	// 将当前3D gaussian的中心点从世界坐标系投影到裁剪坐标系（ray space）
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);  // 想要除以p_hom.w从而转成正常的3D坐标，这里防止除零
	//; 把齐次坐标转换成正常的坐标, 注意这个结果已经在ray spaceg中了
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// Step 2. 从scale和rotation数据中计算3D高斯的协方差矩阵
	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		// 根据当前3D gaussian的尺度和旋转参数计算其对应的协方差矩阵
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Step 3. 计算把3D高斯投影到image space之后得到的2D高斯的两个特征值，就是长短轴的长度
	// 将当前的3D gaussian投影到2D图像，得到对应的2D gaussian的协方差矩阵cov
	// Compute 2D screen-space covariance matrix
	//; 注意：这里的cov是图像平面上的2D高斯的协方差矩阵，所以后续分解出来的长短轴的单位就是像素
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// 计算当前2D gaussian的协方差矩阵cov的逆矩阵，因为协方差矩阵的逆矩阵才反映了2D高斯的长短轴
	// Invert covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;  // 行列式的逆
	// conic是cone的形容词，意为“圆锥的”。猜测这里是指圆锥曲线（椭圆）
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// 计算2D gaussian的协方差矩阵cov的特征值lambda1, lambda2, 从而计算2D gaussian的最大半径
    // 对协方差矩阵进行特征值分解时，可以得到描述分布形状的主轴（特征向量）以及这些轴上分布的宽度（特征值）
	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	// 韦达定理求二维协方差矩阵的特征值
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	// 这里就是截取Gaussian的中心部位（3σ原则），只取像平面上半径为my_radius的部分
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	
	// Step 4.计算当前2D高斯投影到图像上之后，覆盖哪些tile
	// 将归一化设备坐标（Normalized Device Coordinates, NDC）转换为像素坐标
	// ndc2Pix(v, S) = ((v + 1) * S - 1) / 2 = (v + 1) / 2 * S - 0.5
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	// 计算当前的2D gaussian落在哪几个tile上
	//; rect_min(x, y) / rect_max(x, y) 存储当前2D高斯覆盖的左上角和右下角tile的坐标
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	// 如果没有命中任何一个title则直接返回
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;
	
	// Step 5.利用球谐系数计算当前高斯的颜色
	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		//; 注意这里传入3D高斯的world系下中心orig_points、当前相机中心cam_pos, 就相当于包含了视角信息, 可以带入球谐基函数中计算颜色
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Step 6.保存结果到输出变量中
	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;   // 深度，即3D高斯中心点在相机坐标系下的z轴距离
	radii[idx] = my_radius;   // Gaussian在像平面坐标系下的最大半径
	points_xy_image[idx] = point_image;   // Gaussian中心在图像上的像素坐标
	// 前三个将被用于计算高斯的指数部分从而得到 prob（查询点到该高斯的距离->prob，例如，若查询点位于该高斯的中心则 prob 为 1）。最后一个是该高斯本身的密度。
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
// 线程在读取数据（把数据从公用显存拉到block自己的显存）和进行计算之间来回切换，
// 使得线程们可以共同读取Gaussian数据。这样做的原因是block共享内存比公共显存快得多。
template <uint32_t CHANNELS>   // CHANNELS取3，即RGB三个通道
// __launch_bounds__(BLOCK_X * BLOCK_Y)指示编译器在启动内核函数时使用这个特定的线程块大小
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)  
renderCUDA(
	// 使用 __restrict__ 关键字，是为了告诉编译器这些指针之间不存在别名关系，可以进行更有效的优化
	const uint2* __restrict__ ranges,   // 每个tile对应排过序的数组中的索引范围，.x是起始索引, .y是结束索引
	const uint32_t* __restrict__ point_list,  // 按 [tile | 深度] 排序后的Gaussian ID列表
	int W, int H,  // 图像宽高
	const float2* __restrict__ points_xy_image,  // (P, 2), 图像上每个Gaussian中心的2D坐标
	const float* __restrict__ features,   // (P, 3), RGB颜色
	const float4* __restrict__ conic_opacity,   // (P, 4), 每个2D高斯协方差矩阵的逆 + 不透明度
	float* __restrict__ final_T,  // [out], 最终的透光率
	// [out], 多少个Gaussian对该像素的颜色有贡献（用于反向传播时判断各个Gaussian有没有梯度）
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,  // 背景颜色
	float* __restrict__ out_color)  // [out], 渲染结果（图片）
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;  // x方向上tile的个数
	// 当前处理的tile的左上角的像素坐标（我负责的tile的坐标较小的那个角的坐标）
	// group_index()：当前block在grid中的三维索引，相当于blockIdx
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	// 当前处理的tile的右下角的像素坐标（我负责的tile的坐标较大的那个角的坐标）
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	// 当前线程处理的像素坐标（有可能超过图像边界）
	// thread_index()：当前thread在block中的三维索引，相当于threadIdx
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	// 当前处理的像素idl，二维展平成一维后
	uint32_t pix_id = W * pix.y + pix.x;
	// 浮点型像素坐标
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	// 当前线程负责处理的像素是否超过了图像边界
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;  // 如果当前线程处理的像素超过图像边界，则可以认为他已经完成任务了

	// Load start/end range of IDs to process in bit sorted list.
	// 当前处理的tile对应的point_list_keys的起始id和结束id
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	// BLOCK_SIZE = 16 * 16 = 256
	// 我要把任务分成rounds批，每批处理BLOCK_SIZE个Gaussians
	// 每一批，每个线程负责读取一个Gaussian的信息，
	// 所以该block的256个线程每一批就可以读取256个Gaussian的信息
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	// 还有多少3D gaussian需要处理，表示当前tile（block）对应的需要计算的高斯球的数量
	int toDo = range.y - range.x;  // 当前block一共要处理的Gaussian个数

	// 每个线程取一个，并行读数据到 shared memory。然后每个线程都访问该shared memory，读取顺序一致。
	// Allocate storage for batches of collectively fetched data.
	// __shared__: 同一block中的线程共享的内存，共享显存，被同一个block的thread共享的存储
	__shared__ int collected_id[BLOCK_SIZE];  // 记录各线程处理的高斯球的编号
	__shared__ float2 collected_xy[BLOCK_SIZE];  // 记录各线程处理的高斯球中心在2d平面的投影坐标
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];  // 记录各线程处理的高斯球的2d协方差矩阵的逆和不透明度

	// Initialize helper variables
	float T = 1.0f;   // T = transmittance，透射率
	uint32_t contributor = 0;   // 计算当前线程负责的像素一共经过了多少高斯
	uint32_t last_contributor = 0;   // 当前像素渲染时最后一个高斯在当前tile中的索引
	float C[CHANNELS] = { 0 };  // 最后渲染的颜色

	// Iterate over batches until all done or range is complete
	// 当前block(tile)覆盖的所有高斯个数，要256个线程循环拿到共享显存中，直到把所有高斯都处理完
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		// 它首先具有__syncthreads的功能（让所有线程回到同一个起跑线上），并且返回对于多少个线程来说done是true。
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;   // 如果一个block里面全部thread完成，则退出循环
		
		// Step 1.首先256个线程合作把256个高斯球取到 shared memory中
		// Collectively fetch per-Gaussian data from global to shared
		// 由于当前block的线程要处理同一个tile，所以它们面对的Gaussians也是相同的
		// 因此合作读取BLOCK_SIZE条的数据。
		// 之所以分批而不是一次读完可能是因为block的共享内存空间有限。
		// 共同从全局显存读取每一个高斯数据到共享显存，已经结束的thread去取
		int progress = i * BLOCK_SIZE + block.thread_rank();  //thread_rank：当前线程在组内的标号，区间为[0, num_threads)
		if (range.x + progress < range.y)   // 如果当前线程有效，即处理的高斯球不越界
		{
			// point_list表示与已排序的point_list_keys对应的高斯球编号，coll_id为当前线程处理的高斯球编号
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;  //当前线程处理的高斯球id
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];  //points_xy_image:高斯椭圆中心在像素平面的坐标
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];   //2d协方差矩阵的逆和不透明度
		}
		// 等到256个线程把高斯数据都取出来放到 shared memory 中，然后下一步每个线程就负责一个像素的渲染
		block.sync();   // 这下collected_*** 数组就被填满了

		// Step 2.此时256个高斯准备完毕，则每个线程负责一个像素的渲染，对256个高斯挨个计算对像素的渲染贡献
		// Iterate over current batch	
		// 每个线程遍历当前batch一堆高斯球
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// 下面计算当前Gaussian的不透明度
			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];  // 当前处理的2D gaussian在图像上的中心点坐标
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };  // 当前处理的2D gaussian的中心点到当前处理的pixel的offset
			float4 con_o = collected_conic_opacity[j];  // 当前处理的2D gaussian的协方差矩阵的逆矩阵以及它的不透明度
			// 计算高斯分布的强度（或权重），用于确定像素在光栅化过程中的贡献程度
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			// opacity * 像素点出现在这个高斯的几率
			// Gaussian对于这个像素点来说的不透明度
			// 注意con_o.w是”opacity“，是Gaussian整体的不透明度
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)  // 太小了就当成透明的
				continue;
			float test_T = T * (1 - alpha);  // alpha合成的系数，表示经过此高斯之后的光强
			if (test_T < 0.0001f)  // 累乘不透明度到一定的值，标记这个像素的渲染结束
			{
				done = true;   // 表示当前线程负责的像素已经渲染完成
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			// 计算当前Gaussian对像素颜色的贡献
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;
 
			T = test_T;    // 更新透光率

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}
	
	// 所有处理有效像素的thread都会将其最终渲染数据写入帧缓冲区和辅助缓冲区
	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)   // 如果当前线程负责渲染的像素在图像范围内，则把最终渲染的结果写入到输出结果中
	{
		// 用于反向传播计算梯度
		final_T[pix_id] = T;  // 渲染过程后每个像素的最终透明度或透射率值
		// 记录数量，用于提前停止计算
		n_contrib[pix_id] = last_contributor;  // 最后一个贡献的2D gaussian是第几个
		// 把渲染出来的像素值写进out_color
		for (int ch = 0; ch < CHANNELS; ch++)
			// 注意这里C[ch]前面没有(1-T)的系数，因为可以把bg_color[ch]当成最后一个颜色，所以再加上他就是 += T * bg_color[ch]
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
	}
}

// 一个线程负责一个像素，一个block负责一个tile
void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,      // 每个瓦片（tile）在排序后的高斯ID列表中的范围
		point_list,  // 排序后的3D gaussian的id列表
		W, H,   // 图像的宽和高
		means2D,  // 每个2D gaussian在图像上的中心点位置
		colors,   // 每个3D gaussian对应的RGB颜色
		conic_opacity,  // 每个2D gaussian的协方差矩阵的逆矩阵以及它的不透明度
		final_T,  // [out], 渲染过程后每个像素的最终透明度或透射率值
		n_contrib,  // [out], 每个pixel的最后一个贡献的2D gaussian是所在列表范围内的第几个
		bg_color,   // 背景颜色
		out_color); // [out], 输出图像
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	// 这里的写法和AI葵的CUDA教程有一点不一样，这里最外层没有AT_DISPATCH_FLOATING_TYPES
	// 但是后面的<< <(P + 255) / 256, 256 >> >是一样的，是<block, thread>的表示
	// 所以这里其实还是一个kernel函数，真正调用的还是preprocessCUDA函数
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,  // 3D gaussian的个数, 当前激活的球谐函数的次数, 所有球谐系数的个数 
		means3D,  // (P, 3), 每个3D gaussian的XYZ均值
		scales,   // (P, 3), 每个3D gaussian的XYZ尺度
		scale_modifier,  // 尺度缩放系数, 1.0
		rotations,  // (P, 4), 每个3D gaussian的旋转四元组
		opacities,  // (P,), 每个3D gaussian的不透明度
		shs,  // (P, 16, 3), 每个3D gaussian的球谐系数, 用于表示颜色
		clamped,  // (P, 3), 存储每个3D gaussian的R、G、B是否小于0
		cov3D_precomp,  // 提前计算好的每个3D gaussian的协方差矩阵, []
		colors_precomp, // 提前计算好的每个3D gaussian的颜色, []
		viewmatrix,     // (4, 4), 相机外参矩阵, world to camera
		projmatrix,     // (4, 4), 整个投影矩阵，包括W2C和视角变换矩阵
		cam_pos,   // (3,), 相机的中心点XYZ坐标
		W, H,  // 图像的宽和高
		tan_fovx, tan_fovy,  // 水平、垂直视场角一半的正切值
		focal_x, focal_y,  // 水平、垂直方向的焦距
		radii,    // (P,), 存储每个2D gaussian在图像上的半径
		means2D,  // (P, 2), 存储每个2D gaussian的均值
		depths,   // (P,), 存储每个2D gaussian的深度
		cov3Ds,   // (P, 6), 存储每个3D gaussian的协方差矩阵
		rgb,      // (P, 3), 存储每个2D pixel的颜色
		conic_opacity,  // *P, 4), 存储每个2D gaussian的协方差矩阵的逆矩阵以及它的不透明度
		grid,  // dim3, 在水平和垂直方向上需要多少个tile来覆盖整个渲染区域
		tiles_touched,  // (P,), 存储每个2D gaussian覆盖了多少个tile
		prefiltered   // False, 是否预先过滤掉了中心点(均值XYZ)不在视锥（frustum）内的3D gaussian
		);
}