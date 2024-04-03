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

	struct GeometryState
	{
		size_t scan_size;
		float* depths;
		char* scanning_space;
		bool* clamped;
		int* internal_radii;
		float2* means2D;
		float* cov3D;
		float4* conic_opacity;
		float* rgb;
		uint32_t* point_offsets;
		uint32_t* tiles_touched;

		static GeometryState fromChunk(char*& chunk, size_t P);
	};

	// 存储所有3D gaussian的各个参数的结构体
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
		char* size = nullptr;
		T::fromChunk(size, P);
		return ((size_t)size) + 128;
	}
};