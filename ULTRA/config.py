'''
Author: JohnleeHIT 525570844@qq.com
Date: 2022-06-13 16:00:21
LastEditors: JohnleeHIT 525570844@qq.com
LastEditTime: 2022-08-05 09:56:52
FilePath: \ULTRA\src\config.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import torch.backends.cudnn as cudnn

class BaseConfig(object):
    write_log = True
    # checkpoint_path = "result/resnet34_LDL_测试std_pretrain_backbone/runs_std0.01/best_valloss.pt"
    checkpoint_path = ""
    
    # Network setting
    pretrained = True
    # Optimization/150
    num_epoch = 150
    lr_step_size = 60
    lr_ratio = 0.1
    # backbone output features
    num_features = 512


    def __init__(self):
        pass

    def display(self):
        '''
        display configurations
        :return:
        '''
        print("configurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

    def get_str_config(self):
        '''
        get string of the configurations for displaying and storing
        :return: a string of the configuration
        '''
        str_config = "configurations:\n"
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                temp = "{:30} {}\n".format(a, getattr(self, a))
                str_config+=temp
        return str_config

    def write_config(self, log):
        log.write("configurations: \n")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                log.write("{:30} {} \n".format(a, getattr(self, a)))



        log.write("configurations: \n")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                log.write("{:30} {} \n".format(a, getattr(self, a)))
    

class ConfigLDL(BaseConfig):
    # std= 0.02
    # num_discretizing=100
    # num_raters = 3
    # lr_S = 1e-3
    # batchsize = 8
    phase = "validate" # "validate|train|test"
    architecture = "LDL" # "backbone|LDL"
    pretrained_backbone = True
    augmentation = True
    checkpoint_path = ""
    describe = "测试std值"


class ConfigBackbone(BaseConfig):
    # std= 0.02
    # num_discretizing=100
    # num_raters = 3
    # lr_S = 1e-3
    # batchsize = 8
    phase = "validate" # "validate|train|test"
    architecture = "backbone" # "backbone|LDL"
    lr_step_size = 50
    lr_ratio = 0.3
    checkpoint_path = ""
    # checkpoint_path = ""
    describe = "(测试lr=1e-5, 测试MLP+Dropout层)测试单分支LDL, 不加augmentation, 无MSE分支辅助， 更正val数据集错误的弄成了training错误"

   