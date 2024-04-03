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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        """
        parser (ArgumentParser): _description_
        name (str): 参数组的名称
        fill_none (bool, optional): 是否将成员变量的值填充为None
        """
        # 创建一个参数组并将其添加到命令行解析器中
        group = parser.add_argument_group(name)
        # 遍历self对象的所有属性和对应的值
        for key, value in vars(self).items():
            shorthand = False
            # 如果属性名以下划线开头，则为缩写，将属性名去除开头的下划线
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)  # 属性值的类型
            value = value if not fill_none else None 
            # 使用ArgumentParser的add_argument方法添加命令行参数
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        """
        传入所有的参数，从中提取出属于当前类的参数并返回
        """
        group = GroupParams()
        # 遍历 args 对象的所有属性和对应的值，其中 arg 是一个元组，包含属性名和属性值
        # arg[0] 和 arg[1] 分别对应元组中的第一个元素(属性名)和第二个元素(属性值)
        for arg in vars(args).items():
            # 如果参数存在于self对象中，则将命令行参数对象的属性值设置到group对象中
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        # 调用了父类ParamGroup的构造函数
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        # 调用了父类ParamGroup的extract方法，传递了args作为参数，并将返回值赋给了变量g
        g = super().extract(args)
        # 把路径设置为绝对路径
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.random_background = False
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
