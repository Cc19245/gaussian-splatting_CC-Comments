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

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R  # 旋转矩阵
        self.T = T  # 平移向量
        self.FoVx = FoVx   # x方向视场角
        self.FoVy = FoVy   # y方向视场角
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)  # 原始图像
        self.image_width = self.original_image.shape[2]   # 图像宽度
        self.image_height = self.original_image.shape[1]  # 图像高度
 
        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
        
        # 距离相机平面znear和zfar之间且在视锥内的物体才会被渲染
        self.zfar = 100.0   # 最远能看到多远
        self.znear = 0.01   # 最近能看到多近

        self.trans = trans   # 相机中心的平移
        self.scale = scale   # 相机中心坐标的缩放

        # 世界到相机坐标系的变换矩阵，4×4
        #! 疑问：最后又transpose是干嘛的？和glm的列存储有关吗？
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()  
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()  # 投影矩阵
        # bmm是batch matrix multiply, 即批量矩阵乘法，但是这里先unsqueeze扩展一个维度，最后又squeeze压缩这个维度，不知道为啥
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)  # 从世界坐标系到图像的变换矩阵
        self.camera_center = self.world_view_transform.inverse()[3, :3]  # 相机在世界坐标系下的坐标

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

