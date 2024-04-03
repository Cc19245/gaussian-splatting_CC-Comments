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

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:
    """
    Scene 类用于管理场景的3D模型，包括相机参数、点云数据和高斯模型的初始化和加载
    """
    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """
        初始化场景对象，包括：
        1.读取输入数据，包括相机外参、内参、点云，并把结果保存到output文件夹中备用
        2.根据输入的点云数据初始化高斯球，包括每个高斯球的位置、不透明度、颜色等
        
        :param args: 包含模型路径和源路径等模型参数
        :param gaussians: 高斯模型对象，用于场景点的3D表示
        :param load_iteration: 指定加载模型的迭代次数，如果为-1，则自动寻找最大迭代次数
        :param shuffle: 是否在训练前打乱相机列表
        :param resolution_scales: 分辨率比例列表，用于处理不同分辨率的相机
        """
        self.model_path = args.model_path  # 模型文件保存路径
        self.loaded_iter = None  # 已加载的迭代次数
        self.gaussians = gaussians  # 高斯模型对象

        # 检查并加载已有的训练模型
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}  # 用于训练的相机参数
        self.test_cameras = {}   # 用于测试的相机参数

        # 根据数据集类型（COLMAP或Blender）加载场景信息
        # 如果输入数据路径中存在`sparse`这个文件夹，则说明是colmap生成的结果
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            # args.images=images，就是输入数据文件夹下的images文件夹
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        # 如果是初次训练，则把点云、相机信息都保存到ouput文件夹中
        if not self.loaded_iter:
            # ply_path中就存储了点云数据，这里就是把输入点云数据读取出来，然后重新写入到数据文件路径中，并命名为input.ply
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:  # 实测为空
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:   # list, 长度和训练图像的个数一样
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                # 把CameraInfo类型的结果转化成json数据格式，然后存储成cameras.json文件
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        # 这个结果就是一个数字，不知道干嘛的？
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # 根据resolution_scales加载不同分辨率的训练和测试位姿
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        # 如果不是初次训练，则加载已有模型；否则，从点云中初始化3D高斯模型
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            # scene_info.point_cloud就是一个BasicPointCloud类型的点云数据
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        """
        保存当前迭代下的3D高斯模型点云。
        
        :param iteration: 当前的迭代次数。
        """
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        """
        获取指定分辨率比例的训练相机列表
        
        :param scale: 分辨率比例
        :return: 指定分辨率比例的训练相机列表
        """
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]