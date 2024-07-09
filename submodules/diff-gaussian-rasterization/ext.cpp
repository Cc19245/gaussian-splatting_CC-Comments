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

#include <torch/extension.h>
#include "rasterize_points.h"

// 注意：下面绑定的cuda函数是在rasterize_points.h中声明的, 然后这些函数在rasterize_points.cu中定义, 
// 最后rasterize_points.cu中的函数又继续调用了其他.cu文件中的函数。而AI葵的CUDA教程中的函数定义是直接
// 在当前的cppz文件中定义的, 区别不大
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("mark_visible", &markVisible);
}