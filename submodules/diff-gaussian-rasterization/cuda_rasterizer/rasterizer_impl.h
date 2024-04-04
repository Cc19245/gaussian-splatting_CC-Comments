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

#pragma once

#include <iostream>
#include <vector>
#include "rasterizer.h"
#include <cuda_runtime_api.h>

namespace CudaRasterizer
{
	// 在预分配的内存块（chunk）中为不同类型的数组（ptr）分配空间
	// chunk（一个指向当前内存块位置的指针引用），ptr（一个指向分配数组的指针引用），count（数组中元素的数量），alignment（内存对齐要求）
	// 首先计算新的偏移量，以确保 ptr 的对齐，然后更新 chunk 以指向内存块中下一个可用位置
	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		ptr = reinterpret_cast<T*>(offset);
		chunk = reinterpret_cast<char*>(ptr + count);
	}

	// 存储所有3D gaussian的各个参数的结构体
	struct GeometryState
	{
		size_t scan_size;
		float* depths;    // (P,), 每个高斯球的深度
		char* scanning_space;
		bool* clamped;    // (P, 3), 每个高斯球的RGB颜色是否超过[0, 1]范围从而产生截断
		int* internal_radii;   // (P,), 每个高斯球投影到图像上之后的最大半径
		float2* means2D;  // (P, 2), 每个3D高斯球投影成2D高斯球之后的中心位置
		float* cov3D;     // (P, 6), 每个3D高斯球的协方差矩阵
		float4* conic_opacity;   // (P, 4), 每个2D高斯球的协方差矩阵的逆(3个数) + 中心点的不透明度(1个数)
		float* rgb;       // (P, 3), 每个3D高斯球渲染出来的颜色？
		uint32_t* point_offsets;   
		uint32_t* tiles_touched;  // (P,), 存储每个2D gaussian覆盖了多少个tile

		static GeometryState fromChunk(char*& chunk, size_t P);
	};

	struct ImageState
	{
		uint2* ranges;
		uint32_t* n_contrib;
		float* accum_alpha;

		static ImageState fromChunk(char*& chunk, size_t N);
	};

	struct BinningState
	{
		size_t sorting_size;    // 存储用于排序操作的缓冲区大小
		uint64_t* point_list_keys_unsorted;  // 未排序的键列表
		uint64_t* point_list_keys;   // 排序后的键列表
		uint32_t* point_list_unsorted;   // 未排序的点列表
		uint32_t* point_list;   // 排序后的点列表
		char* list_sorting_space;  // 用于排序操作的缓冲区

		static BinningState fromChunk(char*& chunk, size_t P);
	};

	// 计算存储 T 类型数据所需的内存大小的函数
	// 通过调用 T::fromChunk 并传递一个空指针（nullptr）来模拟内存分配过程
	// 通过这个过程，它确定了实际所需的内存大小，加上额外的 128 字节以满足可能的内存对齐要求
	template<typename T> 
	size_t required(size_t P)
	{
		char* size = nullptr;   // 指针从0开始
		T::fromChunk(size, P);  // 分配P个T类型的元素的内存，最终size会指向分配之后下一个可用内存的位置
		// (size_t)sizeb强制把指针转成整数，结果就反映了刚才开辟的内存的大小
		return ((size_t)size) + 128;
	}
};