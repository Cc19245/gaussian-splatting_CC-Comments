#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch.nn as nn
import torch
# 在安装这个包的时候，上一级的setup.py就已经被编译了，然后里面定义的diff_gaussian_rasterization._C就可以被找到了
from . import _C   

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    # 这里就和AI葵讲解的CUDA教程一致，也就是使用torch.autograd.Function包装cuda代码之后，需要显式
    # 调用.apply方法来运算，这样就可以在前向传播之后，调用反向传播计算梯度
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    # staticmethod类似仿函数，不用显式地调用forward这个函数，只通过类名就可以自动调用
    @staticmethod
    def forward(
        ctx,      # context的缩写ctx，保存前向传播时输入的上下文参数，在反向传播的时候计算本地梯度
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg,   # (3,)    [0., 0., 0.]        背景颜色
            means3D,  # (P, 3)  每个3D gaussian的XYZ均值
            colors_precomp,  #  提前计算好的每个3D gaussian的颜色
            opacities,  # (P, 1)  0.1 不透明度
            scales,     # (P, 3)  每个3D gaussian的XYZ尺度
            rotations,  # (P, 4)  [1., 0., 0., 0.]    每个3D gaussian的旋转四元数
            raster_settings.scale_modifier,  # 1.0
            cov3Ds_precomp,  #提前计算好的每个3D gaussian的协方差矩阵
            raster_settings.viewmatrix,  # (4, 4)  相机外参矩阵 world to camera
            raster_settings.projmatrix,  # (4, 4)  相机内参矩阵 camera to image
            raster_settings.tanfovx,  # 0.841174315841308   水平视场角一半的正切值
            raster_settings.tanfovy,  # 0.4717713779864031   垂直视场角一半的正切值
            raster_settings.image_height,  # 546  图像高度
            raster_settings.image_width,   # 979   图像宽度
            sh,  # (P, 16, 3) 每个3D gaussian对应的球谐系数, R、G、B通道分别对应16个球谐系数
            raster_settings.sh_degree,  # 0~3，球谐函数的次数, 最开始是0, 每隔1000次迭代, 将球谐函数的次数增加1
            raster_settings.campos,  # (3,) [-3.9848, -0.3486,  0.1481]  所有相机的中心点坐标
            raster_settings.prefiltered,   # False
            raster_settings.debug   # False
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            # 这个函数对应的是CUDA中的 RasterizeGaussiansCUDA 函数，它们之间的对应关系在ext.cpp文件的PYBIND11_MODULE中声明
            num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer)
        return color, radii

    @staticmethod
    def backward(ctx, grad_out_color, _):
        """
        梯度反向传播的函数，注意输入参数（除了ctx之外）的个数和forward函数最终返回的变量个数相等
        grad_out_color: 上游梯度对 color 的梯度
        _: 用不到，上游梯度对 radii 的梯度
        """
        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color, 
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
            # 实际调用的是 rasterize_points.cu 文件中的 RasterizeGaussiansBackwardCUDA 函数
            grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = \
                _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        """
        初始化光栅化器实例。
        
        :param raster_settings: 光栅化的设置参数，包括图像高度、宽度、视场角、背景颜色、视图和投影矩阵等。
        """
        super().__init__()
        self.raster_settings = raster_settings  # 保存光栅化的设置参数

    def markVisible(self, positions):
        """
        标记给定位置的点是否在相机的视锥体内，即判断点是否对相机可见（基于视锥剔除）。
        
        :param positions: 点的位置，通常是3D高斯分布的中心位置。
        :return: 一个布尔张量，表示每个点是否可见。
        """
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():  # 不计算梯度，因为这一步只是用于判断可见性
            raster_settings = self.raster_settings
            # 调用一个C++/CUDA实现的函数来快速计算可见性
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
        """
        光栅化器的前向传播方法，用于将3D高斯分布渲染成2D图像。

        :param means3D: 3D高斯分布的中心位置。
        :param means2D: 屏幕空间中3D高斯分布的预期位置，用于梯度回传。
        :param opacities: 高斯分布的不透明度。
        :param shs: 球谐系数，用于从方向光照计算颜色。
        :param colors_precomp: 预计算的颜色。
        :param scales: 高斯分布的尺度参数。
        :param rotations: 高斯分布的旋转参数。
        :param cov3D_precomp: 预先计算的3D协方差矩阵。
        :return: 光栅化后的2D图像。
        """
        raster_settings = self.raster_settings

        # 检查输入参数的有效性
        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        # 如果相关参数未提供，则初始化为空张量
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        # 调用C++/CUDA实现的光栅化函数来渲染图像
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings, 
        )

