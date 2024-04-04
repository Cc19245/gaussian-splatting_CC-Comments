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

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>

// 创建并返回一个 lambda 表达式，该表达式用于调整 torch::Tensor 对象的大小，并返回一个指向它数据的原始指针
std::function<char *(size_t N)> resizeFunctional(torch::Tensor &t)
{
	auto lambda = [&t](size_t N)
	{
		t.resize_({(long long)N});
		return reinterpret_cast<char *>(t.contiguous().data_ptr());
	};
	return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor &background,
	const torch::Tensor &means3D,   // (P, 3), 高斯中心位置
	const torch::Tensor &colors,    // None，提前计算好的颜色
	const torch::Tensor &opacity,   // (P, 1)
	const torch::Tensor &scales,    // (P, 3)
	const torch::Tensor &rotations, // (P, 4)
	const float scale_modifier,     // 尺度修改系数
	const torch::Tensor &cov3D_precomp,  // None, 提前计算好的协方差矩阵
	const torch::Tensor &viewmatrix,     // (4, 4), W2C
	const torch::Tensor &projmatrix,     // (4, 4), 整个投影矩阵，包括W2C和视角变换矩阵
	const float tan_fovx,
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const torch::Tensor &sh,   // (P, 16, 3), 所有的球谐系数
	const int degree,    // 当前激活的球谐次数
	const torch::Tensor &campos,   // (3,), 相机中心位置
	const bool prefiltered,
	const bool debug)
{
	if (means3D.ndimension() != 2 || means3D.size(1) != 3)
	{
		AT_ERROR("means3D must have dimensions (num_points, 3)");
	}

	const int P = means3D.size(0);
	const int H = image_height;
	const int W = image_width;
	
	// 数据类型，分别为int和float32
	auto int_opts = means3D.options().dtype(torch::kInt32);
	auto float_opts = means3D.options().dtype(torch::kFloat32);

	// Step 1.定义最终要输出的数据，然后调用cuda的函数之后， 结果就被写入这些数据的缓存中了，最后就可以直接返回了
	// (3, H, W), 在指定的视角下, 对所有3D gaussian进行投影和渲染得到的图像
	torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
	// (P,), 表示每个高斯投影到图像上的最大半径
	torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));

	torch::Device device(torch::kCUDA);
	torch::TensorOptions options(torch::kByte);
	// (0,), 存储所有3D gaussian对应的参数(均值、尺度、旋转参数、不透明度)的tensor, 会动态分配存储空间
	torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
	torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
	// (0,), 存储在指定视角下渲染得到的图像的tensor, 会动态分配存储空间
	torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
	// 动态调整 geomBuffer 大小的函数, 并返回对应的数据指针
	std::function<char *(size_t)> geomFunc = resizeFunctional(geomBuffer);
	// 动态调整 binningBuffer 大小的函数, 并返回对应的数据指针
	std::function<char *(size_t)> binningFunc = resizeFunctional(binningBuffer);
	// 动态调整 imgBuffer 大小的函数, 并返回对应的数据指针
	std::function<char *(size_t)> imgFunc = resizeFunctional(imgBuffer);

	int rendered = 0;
	if (P != 0)   // 有3D高斯球，才需要进行渲染
	{
		int M = 0;   // 球谐基函数的个数，代码实际配置是16
		// .size相当于pytorch中的.shape, 这里就是获取第0维的大小
		if (sh.size(0) != 0)
		{
			M = sh.size(1);
		}

		rendered = CudaRasterizer::Rasterizer::forward(
			geomFunc,
			binningFunc,
			imgFunc,
			P, degree, M,							  // 3D gaussian的个数, 当前激活的球谐函数的次数, 球谐系数的个数 (球谐系数用于表示颜色)
			background.contiguous().data<float>(),	  // 背景颜色, [0, 0, 0]
			W, H,									  // 图像的宽和高
			means3D.contiguous().data<float>(),		  // (P, 3), 每个3D gaussian的XYZ均值
			sh.contiguous().data_ptr<float>(),		  // (P, 16, 3), 每个3D gaussian的球谐系数, 用于表示颜色
			colors.contiguous().data<float>(),		  // None, 提前计算好的每个3D gaussian的颜色, []
			opacity.contiguous().data<float>(),		  // (P, 1), 每个3D gaussian的不透明度
			scales.contiguous().data_ptr<float>(),	  // (P, 3), 每个3D gaussian的XYZ尺度
			scale_modifier,							  // 尺度缩放系数, 1.0
			rotations.contiguous().data_ptr<float>(), // (P, 4)< 每个3D gaussian的旋转四元组
			cov3D_precomp.contiguous().data<float>(), // None, 提前计算好的每个3D gaussian的协方差矩阵, []
			viewmatrix.contiguous().data<float>(),	  // (4, 4), 相机外参矩阵, world to camera
			projmatrix.contiguous().data<float>(),	  // (4, 4), 整个投影矩阵，包括W2C和视角变换矩阵
			campos.contiguous().data<float>(),		  // (3, ), 相机的中心点XYZ坐标
			tan_fovx,								  // 水平视场角一半的正切值
			tan_fovy,								  // 垂直视场角一半的正切值
			prefiltered,							  // False, 是否预先过滤掉了中心点(均值XYZ)不在视锥（frustum）内的3D gaussian, False
			out_color.contiguous().data<float>(),	  // (3, H, W), 输出结果, 在指定的视角下, 对所有3D gaussian进行投影和渲染得到的图像
			radii.contiguous().data<int>(),			  // (P, ), 输出结果，存储每个2D gaussian在图像上的半径
			debug);									  // False
	}

	return std::make_tuple(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansBackwardCUDA(
	const torch::Tensor &background,
	const torch::Tensor &means3D,
	const torch::Tensor &radii,
	const torch::Tensor &colors,
	const torch::Tensor &scales,
	const torch::Tensor &rotations,
	const float scale_modifier,
	const torch::Tensor &cov3D_precomp,
	const torch::Tensor &viewmatrix,
	const torch::Tensor &projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const torch::Tensor &dL_dout_color,
	const torch::Tensor &sh,
	const int degree,
	const torch::Tensor &campos,
	const torch::Tensor &geomBuffer,
	const int R,
	const torch::Tensor &binningBuffer,
	const torch::Tensor &imageBuffer,
	const bool debug)
{
	const int P = means3D.size(0);
	const int H = dL_dout_color.size(1);
	const int W = dL_dout_color.size(2);

	int M = 0;
	if (sh.size(0) != 0)
	{
		M = sh.size(1);
	}

	torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
	torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
	torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
	torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
	torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
	torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
	torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
	torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
	torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());

	if (P != 0)
	{
		CudaRasterizer::Rasterizer::backward(P, degree, M, R,
											 background.contiguous().data<float>(),
											 W, H,
											 means3D.contiguous().data<float>(),
											 sh.contiguous().data<float>(),
											 colors.contiguous().data<float>(),
											 scales.data_ptr<float>(),
											 scale_modifier,
											 rotations.data_ptr<float>(),
											 cov3D_precomp.contiguous().data<float>(),
											 viewmatrix.contiguous().data<float>(),
											 projmatrix.contiguous().data<float>(),
											 campos.contiguous().data<float>(),
											 tan_fovx,
											 tan_fovy,
											 radii.contiguous().data<int>(),
											 reinterpret_cast<char *>(geomBuffer.contiguous().data_ptr()),
											 reinterpret_cast<char *>(binningBuffer.contiguous().data_ptr()),
											 reinterpret_cast<char *>(imageBuffer.contiguous().data_ptr()),
											 dL_dout_color.contiguous().data<float>(),
											 dL_dmeans2D.contiguous().data<float>(),
											 dL_dconic.contiguous().data<float>(),
											 dL_dopacity.contiguous().data<float>(),
											 dL_dcolors.contiguous().data<float>(),
											 dL_dmeans3D.contiguous().data<float>(),
											 dL_dcov3D.contiguous().data<float>(),
											 dL_dsh.contiguous().data<float>(),
											 dL_dscales.contiguous().data<float>(),
											 dL_drotations.contiguous().data<float>(),
											 debug);
	}

	return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations);
}

torch::Tensor markVisible(
	torch::Tensor &means3D,
	torch::Tensor &viewmatrix,
	torch::Tensor &projmatrix)
{
	const int P = means3D.size(0);

	torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));

	if (P != 0)
	{
		CudaRasterizer::Rasterizer::markVisible(P,
												means3D.contiguous().data<float>(),
												viewmatrix.contiguous().data<float>(),
												projmatrix.contiguous().data<float>(),
												present.contiguous().data<bool>());
	}

	return present;
}