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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

class GaussianModel:

    def setup_functions(self):
        """
        定义和初始化一些用于处理3D高斯模型参数的函数。
        """

        # 定义构建3D高斯协方差矩阵的函数
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            # 注意这里是带有一个batch维度的, 所以最终维度是(B, 3, 3)
            actual_covariance = L @ L.transpose(1, 2)  # 计算实际的协方差矩阵
            symm = strip_symmetric(actual_covariance)  # 提取协方差矩阵中的对称元素, 剩下的元素和这些相关
            return symm
        
        # 初始化一些激活函数
        self.scaling_activation = torch.exp  # 用exp函数确保尺度参数非负
        self.scaling_inverse_activation = torch.log   # 尺度参数的逆激活函数, 用于梯度回传

        # 协方差矩阵的激活函数
        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid   # 用sigmoid函数确保不透明度在0到1之间
        self.inverse_opacity_activation = inverse_sigmoid  # 不透明度的逆激活函数

        # 用于标准化旋转参数的函数, 因为旋转参数是用四元数表示的, 而只有单位四元数才能表示旋转
        self.rotation_activation = torch.nn.functional.normalize  


    def __init__(self, sh_degree : int):
        """
        初始化3D高斯模型的参数。

        :param sh_degree: 球谐函数的最大次数, 用于控制颜色表示的复杂度。
        """
        # 初始化球谐次数和最大球谐次数
        self.active_sh_degree = 0   # 目前的球谐阶数, 在`oneupSHdegree()`方法中加一, SH = Sphere Harmonics
        self.max_sh_degree = sh_degree  # 允许的最大球谐次数

        # 初始化3D高斯模型的各项参数, torch.empty(0) 创建一个形状为空的张量, 占用很小的内存空间
        self._xyz = torch.empty(0)  # 3D高斯的中心位置 (均值) 
        self._features_dc = torch.empty(0)  # # 球谐的直流分量 (dc = Direct Current) 
        self._features_rest = torch.empty(0)  # 其余的球谐系数, 用于表示颜色的细节和变化
        self._scaling = torch.empty(0)    # 3D高斯的尺度参数, 控制高斯的宽度
        self._rotation = torch.empty(0)   # 3D高斯的旋转参数, 用四元数表示
        self._opacity = torch.empty(0)    # 3D高斯的不透明度 (经历sigmoid前的) , 控制可见性
        self.max_radii2D = torch.empty(0) # 在某个相机视野里出现过的 (像平面上的) 最大2D半径, 详见train.py里面gaussians.max_radii2D[visibility_filter] = ...一行
        self.xyz_gradient_accum = torch.empty(0)  # 用于累积3D高斯中心位置的梯度
        self.denom = torch.empty(0)  # 与累积梯度配合使用, 表示统计了多少次累积梯度, 算平均梯度时除掉这个 (denom = denominator, 分母) 
        self.optimizer = None  # 优化器, 用于调整上述参数以改进模型
        self.percent_dense = 0  # 参与控制Gaussian密集程度的超参数
        self.spatial_lr_scale = 0  # 坐标的学习率要乘上这个, 抵消在不同尺度下应用同一个学习率带来的问题

        # 调用setup_functions来g定义各种变量的激活函数、协方差矩阵的生成方式等
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        # (B, 16, 3)
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        """
        从点云数据初始化模型参数。

        :param pcd: 点云数据, 包含点的位置和颜色。
        :param spatial_lr_scale: 空间学习率缩放因子, 影响位置参数的学习率。
        """

        '''
        根据scene.Scene.__init__以及
		scene.dataset_readers.SceneInfo.nerf_normalization, 
		即scene.dataset_readers.getNerfppNorm的代码, 
		这个值似乎是训练相机中离它们的坐标平均值 (即中心) 最远距离的1.1倍, 
		根据命名推断应该与学习率有关, 防止固定的学习率适配不同尺度的场景时出现问题。
        '''
        self.spatial_lr_scale = spatial_lr_scale

        # 将点云的位置和颜色数据从numpy数组转换为PyTorch张量, 并传送到CUDA设备上
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        
        # 把RGB颜色信息转成球谐系数的直流分量C0
        '''
        应为球谐的直流分量, 大小为(N, 3)
		RGB2SH(x) = (x - 0.5) / 0.28209479177387814
		看样子pcd.colors的原始范围应该是0到1。
		0.28209479177387814是1 / (2*sqrt(pi)), 是直流分量Y(l=0,m=0)的值。
        '''
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        
        # 初始化存储球谐系数的张量, 每个颜色通道有(max_sh_degree + 1) ** 2个球谐系数
        # RGB三通道球谐的所有系数, 大小为(N, 3, (最大球谐阶数 + 1)²)
        # 这里维度就是(B, 3, 16), 其中B是所有高斯球的个数, 3是RGB三个通道, 16是从0-max_sh_degree所有球谐基函数的个数
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()  # (P, 3, 16)
        features[:, :3, 0 ] = fused_color  # 将RGB转换后的球谐系数C0项的系数存入
        #! 疑问：这么写访问不到3:吧？
        features[:, 3:, 1:] = 0.0   # 其余球谐系数初始化为0

        # 打印初始点的数量
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        
        # 计算点云中每个点到其最近的k个点的平均距离的平方, 用于确定高斯的尺度参数
        '''
        dist2的大小应该是(N,)。
		首先可以明确的是这句话用来初始化scale, 且scale (的平方) 不能低于1e-7。
		我阅读了一下submodules/simple-knn/simple_knn.cu, 大致猜出来了这个是什么意思。
		(cu文件里面一句注释都没有, 读起来真折磨！)
		distCUDA2函数由simple_knn.cu的SimpleKNN::knn函数实现。
		KNN意思是K-Nearest Neighbor, 即求每一点最近的K个点。
		simple_knn.cu中令k=3, 求得每一点最近的三个点距该点的平均距离。
		算法并没有实现真正的KNN, 而是近似KNN。
		原理是把3D空间中的每个点用莫顿编码 (Morton Encoding) 转化为一个1D坐标
		 (详见https://www.fwilliams.info/point-cloud-utils/sections/morton_coding/, 
		用到了能够填满空间的Z曲线) , 
		然后对1D坐标进行排序, 从而确定离每个点最近的三个点。
		simple_knn.cu实际上还用了一种加速策略, 是将点集分为多个大小为1024的块 (box) , 
		在每个块内确定3个最近邻居和它们的平均距离。用平均距离作为Gaussian的scale。
		 (我的解读不一定准确, 如有错误请指正) 
        '''
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        
        '''
		因为scale的激活函数是exp, 所以这里存的也不是真的scale, 而是ln(scale)。
		注意dist2其实是距离的平方, 所以这里要开根号。
		repeat(1, 3)标明三个方向上scale的初始值是相等的。
		scales的大小：(N, 3)
        '''
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        
        # 旋转矩阵, 大小为(N, 4), 初始化每个点的旋转参数为单位四元数 (无旋转) 
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        # 不透明度在经历sigmoid前的值, 大小为(N, 1)。初始化每个点的不透明度为0.1 (通过inverse_sigmoid转换) 
        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))  # (P, 1)

        # 将以上计算的参数设置为模型的可训练参数
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))  # 位置
        # (B, 1, 3)
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))   # 球谐系数C0项
        # (B, 15, 3)
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))  # 其余球谐系数
        self._scaling = nn.Parameter(scales.requires_grad_(True))  # 尺度
        self._rotation = nn.Parameter(rots.requires_grad_(True))   # 旋转
        self._opacity = nn.Parameter(opacities.requires_grad_(True))  # 不透明度
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")  # 存储2D投影的最大半径, 初始化为0

    def training_setup(self, training_args):
        """
        设置训练参数, 包括初始化用于累积梯度的变量, 配置优化器, 以及创建学习率调度器

        :param training_args: 包含训练相关参数的对象。
        """
        # 设置在训练过程中, 用于密集化处理的3D高斯点的比例
        self.percent_dense = training_args.percent_dense
        # 初始化用于累积3D高斯中心点位置梯度的张量, 用于之后判断是否需要对3D高斯进行克隆或切分
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # 配置各参数的优化器, 包括指定参数、学习率和参数名称
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        # 创建优化器, 这里使用Adam优化器
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        # 创建学习率调度器, 用于对中心点位置的学习率进行调整
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        """
        根据当前的迭代次数动态调整xyz参数的学习率
        
        :param iteration: 当前的迭代次数。
        """
        ''' Learning rate scheduling per step '''
        # 遍历优化器中的所有参数组
        for param_group in self.optimizer.param_groups:
            # 找到名为"xyz"的参数组, 即3D高斯分布中心位置的参数
            if param_group["name"] == "xyz":
                # 使用xyz_scheduler_args函数 (一个根据迭代次数返回学习率的调度函数) 计算当前迭代次数的学习率
                lr = self.xyz_scheduler_args(iteration)
                # 将计算得到的学习率应用到xyz参数组
                param_group['lr'] = lr
                # 返回这个新的学习率值, 可能用于日志记录或其他用途
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        """
        重置不透明度参数。这个方法将所有的不透明度值设置为一个较小的值 (但不是0) , 以避免在训练过程中因为不透明度过低而导致的问题。
        """
        # get_opacity返回了经过exp的不透明度, 是真的不透明度
        # 这句话让所有不透明度都不能超过0.01
        # 使用inverse_sigmoid函数确保新的不透明度值在适当的范围内, 即使它们已经很小 (最小设定为0.01) 
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        # 更新优化器中不透明度参数的值
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        # 将更新后的不透明度参数保存回模型中
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        """
        将指定的参数张量替换到优化器中, 这主要用于更新模型的某些参数 (例如不透明度) 并确保优化器使用新的参数值。

        :param tensor: 新的参数张量。
        :param name: 参数的名称, 用于在优化器的参数组中定位该参数。
        :return: 包含已更新参数的字典。
        """
        # 看样子是把优化器保存的某个名为`name`的参数的值强行替换为`tensor`
    	# 这里面需要注意的是修改Adam优化器的状态变量：动量 (momentum) 和平方动量 (second-order momentum) 
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # 定位到指定名称的参数组
            if group["name"] == name:
                # 获取并重置优化器状态 (动量等) 
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)  # 重置一阶动量
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)  # 重置二阶动量

                # 删除旧的优化器状态, 并使用新的参数张量创建一个新的可优化参数
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                # 保存更新后的参数
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        """
        删除不符合要求的3D高斯分布在优化器中对应的参数

        :param mask: 一个布尔张量, 表示需要保留的3D高斯分布。
        :return: 更新后的可优化张量字典。
        """
        # 根据`mask`裁剪一部分参数及其动量和二阶动量
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                # 更新优化器状态
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                # 删除旧状态并更新参数
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        """
        删除不符合要求的3D高斯分布。

        :param mask: 一个布尔张量, 表示需要删除的3D高斯分布。
        """
        # 删除Gaussian并移除对应的所有属性
        # 生成有效点的掩码并更新优化器中的参数
        valid_points_mask = ~mask  # 有效的高斯点的标志位
        # 从优化器中只保留有效的高斯球的相关参数
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        # 重新从优化器中读取这些参数到类成员变量中, 结果就相当于删掉了不需要的高斯球
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        
        # 更新累积梯度和其他相关张量
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        """
        将新的参数张量添加到优化器的参数组中
        """
        # 把新的张量字典添加到优化器
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        """
        将新生成的3D高斯分布的属性添加到模型的参数中。
        """
        # 新增Gaussian, 把新属性添加到优化器中
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        # 把新增的高斯加入到优化器中
        optimizable_tensors = self.cat_tensors_to_optimizer(d)

        # 从优化器中重新读取参数, 结果重新赋值给类的成员变量, 这样就完成了新高斯的添加
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        """
        对那些梯度超过一定阈值且尺度大于一定阈值的3D高斯进行分割操作。
        这意味着这些高斯可能过于庞大, 覆盖了过多的空间区域, 需要分割成更小的部分以提升细节。
        """
        '''
		被分裂的Gaussians满足两个条件：
		1.  (平均) 梯度过大；
		2. 在某个方向的最大缩放大于一个阈值。
		参照论文5.2节“On the other hand...”一段, 大Gaussian被分裂成两个小Gaussians, 
		其放缩被除以φ=1.6, 且位置是以原先的大Gaussian作为概率密度函数进行采样的。
		'''
        # 初始化
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        # 选择满足条件的点
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        # 计算新高斯分布的属性
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)  # 尺度
        means =torch.zeros((stds.size(0), 3),device="cuda")  # 均值 (新分布的中心点) 
        samples = torch.normal(mean=means, std=stds)  # 随机采样新的位置
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)  # 旋转
        
        # 计算新的位置
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        # 调整尺度并保持其他属性
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        # 将分割得到的新高斯分布的属性添加到模型中
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        # 删除原有过大的高斯分布
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        """
        对那些梯度超过一定阈值且尺度小于一定阈值的3D高斯进行克隆操作。
        这意味着这些高斯在空间中可能表示的细节不足, 需要通过克隆来增加细节。
        """
        # Extract points that satisfy the gradient condition
        # 提取出大于阈值`grad_threshold`且缩放参数小于self.percent_dense * scene_extent的Gaussians, 在下面进行克隆
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        # 提取这些点的属性
        new_xyz = self._xyz[selected_pts_mask]  # 位置
        new_features_dc = self._features_dc[selected_pts_mask]      # 直流分量 (基本颜色) 
        new_features_rest = self._features_rest[selected_pts_mask]  # 其他球谐分量
        new_opacities = self._opacity[selected_pts_mask]  # 不透明度
        new_scaling = self._scaling[selected_pts_mask]    # 尺度
        new_rotation = self._rotation[selected_pts_mask]  # 旋转

        # 将克隆得到的新高斯分布的属性添加到模型中
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        """
        对3D高斯分布进行密集化和修剪的操作

        :param max_grad: 梯度的最大阈值, 用于判断是否需要克隆或分割。
        :param min_opacity: 不透明度的最小阈值(0.005), 低于此值的3D高斯将被删除。
        :param extent: 场景的尺寸范围, 用于评估高斯分布的大小是否合适。
        :param max_screen_size: 最大屏幕尺寸阈值, 用于修剪过大的高斯分布。
        """
        # 计算3D高斯中心的累积梯度并修正NaN值
        grads = self.xyz_gradient_accum / self.denom  # 计算平均梯度
        grads[grads.isnan()] = 0.0

        # 根据梯度和场景范围阈值进行克隆或分割操作
        self.densify_and_clone(grads, max_grad, extent)   # 通过克隆增加密度
        self.densify_and_split(grads, max_grad, extent)   # 通过分裂增加密度

        # 接下来移除一些Gaussians, 它们满足下列要求中的一个：
		# 1. 接近透明 (不透明度小于min_opacity) 
		# 2. 在某个相机视野里出现过的最大2D半径大于屏幕 (像平面) 大小
		# 3. 在某个方向的最大缩放大于0.1 * extent (也就是说很长的长条形也是会被移除的) 
        # 创建修剪掩码以删除不必要的3D高斯分布
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        
        # 删除不符合要求的3D高斯分布
        self.prune_points(prune_mask)

        # 释放已分配但未使用的 GPU 存储空间, 因为前面进行了删除3D高斯的操作, 这里就释放缓存
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        # 统计2D高斯的像素坐标的累积梯度和均值的分母 (即迭代步数？) 
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1