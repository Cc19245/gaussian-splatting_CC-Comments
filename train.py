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
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    """

    Args:
        dataset (_type_): 模型相关的参数
        opt (_type_): 优化相关的参数
        pipe (_type_): 训练流程相关的参数
        testing_iterations (_type_): _description_
        saving_iterations (_type_): _description_
        checkpoint_iterations (_type_): _description_
        checkpoint (_type_): _description_
        debug_from (_type_): _description_
    """
    first_iter = 0
    # 新建保存模型输出结果的文件夹，并且把配置参数保存下来
    tb_writer = prepare_output_and_logger(dataset)

    # 初始化高斯模型，实际上就是定义高斯球的各个参数（位置、不透明度、颜色等）的激活函数
    gaussians = GaussianModel(dataset.sh_degree)

    # 加载数据集和对应的相机参数，然后根据输入的点云初始化场景（即初始化高斯球的位置、不透明度、颜色等）
    scene = Scene(dataset, gaussians)

    # 为高斯模型参数设置优化器和学习率调度器
    gaussians.training_setup(opt)

    # 如果提供了checkpoint，则从checkpoint加载模型参数并恢复训练进度
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # 设置背景颜色，白色或黑色取决于数据集要求
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 创建CUDA事件用于计时
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    # 使用tqdm库创建进度条，追踪训练进度
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        # 下面这段代码应该和训练无关，是在训练过程中打开gui显示，然后可以用当前训练出来的模型实时地进行渲染，从而反映当前训练效果
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        # 记录迭代开始时间
        iter_start.record()

        # 根据当前迭代次数更新学习率
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # 每1000次迭代，提升球谐函数的次数以改进模型复杂度
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        # 随机选择一个训练用的相机视角
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        # 这里pop就是从指定分辨率的所有相机list中，随机弹出（randint）一个相机，使用它进行训练
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        # 如果达到调试起始点，启用调试模式
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # 根据设置决定是否使用随机背景颜色
        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # Step 重点：渲染当前视角的图像
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        # image: (3, H, W) 渲染得到的图像
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        # 计算渲染图像与真实图像之间的损失
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        # 记录迭代结束时间
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            # 更新进度条和损失显示
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                # set_postfix 是一个用于设置进度条显示后缀信息的方法
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            # 定期记录训练数据并保存模型
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            # 在指定迭代区间内，对3D高斯模型进行增密和修剪
            if iteration < opt.densify_until_iter:  # 默认参数为15_000
                # Keep track of max radii in image-space for pruning 
                # 在图像空间跟踪最大半径，以便进行修剪操作
                # 更新高斯球投影到图像上的最大半径
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                # 累积2D高斯中心的梯度值和跟踪次数
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                # densify_from_iter: 默认500， densification_interval: 默认100
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    # opacity_reset_interval: 默认3000
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    # densify_grad_threshold: 默认0.0002，也就是累积梯度超过这个阈值之后，就需要对高斯球进行增密和修剪
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            # 执行优化器的一步，并准备下一次迭代
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            # 定期保存checkpoint
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    # 如果没有指定模型输出路径，则自定义一个模型路径
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())  # 使用 uuid.uuid4() 函数生成一个唯一的字符串，并将其转换为字符串格式
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    # 把当前的配置参数写入文件中
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        # 将命令行参数对象 args 转换为字典；使用 Namespace 类将字典转换为命名空间对象；将命名空间对象转换为字符串形式
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    # 注意把parser传入下面的三个类之后，parser中就会增加很多类中定义的参数
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    # store_true 是一个动作，表示当命令行中指定了 --quiet 参数时，将其值设为 True；否则其值为 False。
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    # sys.argv[1:] 是一个列表，表示从命令行输入的参数列表，去掉了第一个元素（通常是脚本的名称）
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    # 这里 model_path 就是在 ModelParams 类中增加的
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    # 这里主要的作用是设置随机种子
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    # 是否开启 PyTorch 的自动异常检测功能。这个功能在调试深度学习模型时非常有用，
    # 它可以帮助发现梯度计算过程中的数值问题或潜在的数值不稳定性，从而提高模型的可靠性和稳定性
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
