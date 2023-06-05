
import torch
from torch.optim import Adam, lr_scheduler
import torch.utils.data as Data
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import csv
from datetime import datetime
import warnings
import argparse
import glob
from utils import l2_regularisation, unimodal_loss
from dataset import TrainBoneAge,TestBoneAge
from net import ULTRA, Classifier, Fusion, Baseline
from metrics import *


warnings.filterwarnings("ignore")


def parse_args():
    """
    dist_mode: 标签生成是用guassian还是t_lgd
    use_emf: 是否用sigmoid自监督
    use_fdmrm: 是否用vae
    n_rater: 分支数目
    weight内不同维度代表不同loss的权重，0:vae_loss,1:kl_loss,2:mse_loss,3:unimodal_loss
    例：[1, 1, 1, 0]代表使用了vae_loss,mse_loss,kl_loss(vae_loss只有在use_fdmrm=True时才参与训练，其他时候vae_loss=0)
    weight_cache = [[1, 0, 1, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]]
    weight = weight_cache[0] 只用mse
    weight = weight_cache[1] 只用kl
    weight = weight_cache[2] kl+mse
    weight = weight_cache[3] kl+mse+unimodal
    """
    parser = argparse.ArgumentParser(description='Test SLDL')
    parser.add_argument('--experiment', type=str, help='实验标记', default='test')
    parser.add_argument('--gpu', type=int, help='gpu序号', default=0)
    parser.add_argument('--dist_mode', type=str, help='标签生成是用guassian还是tlgd', default="guassian")
    parser.add_argument('--use_emf', type=str, help='是否用sigmoid自监督', default="True")
    parser.add_argument('--use_fdmrm', type=str, help='是否用vae', default="True")
    parser.add_argument('--n_rater', type=int, help='分支数目', default=1)
    parser.add_argument('--sigma', type=float, help='温度系数', default=0.1)
    parser.add_argument('--weight', nargs='+', help='不同loss权重', default=[0.1, 0, 1, 0.1])
    parser.add_argument('--data_path', type=str, help='数据路径', default="/home/lxj/bone_age")
    parser.add_argument('--log_path', type=str, help='训练日志路径', default="./log")
    parser.add_argument('--checkpoints_path', type=str, help='模型保存路径', default="./checkpoints")
    parser.add_argument('--result_path', type=str, help='预测结果保存路径', default="./result")
    parser.add_argument('--pretrain_model_path', type=str, help='预训练模型路径', default="./pretrain_model/pretrain.pt")
    args = parser.parse_args()
    return args


args = parse_args()
experiment = str(args.experiment)
gpu = args.gpu
dist_mode = args.dist_mode
use_emf = args.use_emf == "True"
use_fdmrm = args.use_fdmrm == "True"
n_rater = args.n_rater
sigma = args.sigma
weight_ = args.weight
weight = [float(i) for i in weight_]
device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')
print(experiment, gpu, dist_mode, use_emf, use_fdmrm, n_rater, sigma, weight)
data_path = args.data_path

pretrain_model_path = args.pretrain_model_path
current_time = datetime.now().strftime('%b%d_%H-%M-%S')
result_path = args.result_path
num_classes = 13  # 使用tlgd生成的标签分布维度为100
if dist_mode == "guassian":  # 使用guassian生成的标签分布维度
    num_classes = 250
num_filters = 128  # eifg生成的特征向量的维度
latent_dim = 6  # esfg生成的隐空间的特征向量的维度
out_features = num_classes
image=os.path.join(data_path, "boneage-training-dataset/boneage-training-dataset/*.png")
imgdir = glob.glob(image)
label=os.path.join(data_path, 'boneage-training-dataset.csv')
test_set = TestBoneAge(n_raters=n_rater, sigma=sigma, dist_mode=dist_mode,
                          num_classes=num_classes, image=imgdir[-2000:],
                          label=os.path.join(data_path, 'boneage-training-dataset.csv'))
test_dataloader = Data.DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True)
print("---data loaded---")
net = Baseline(out_features).to(device)
if use_fdmrm:
    net = ULTRA(in_features=256, out_features=out_features, num_classes=num_classes, num_filters=num_filters, latent_dim=latent_dim,
                n_raters=n_rater, training=True, device=device)
classifier = Classifier(in_features=out_features , num_classes=num_classes).to(device)
fusion = Fusion().to(device)
optimizer = Adam([*net.parameters()] + [*classifier.parameters()], lr=1e-4, weight_decay=1e-5,betas=(0.97, 0.999))

pretrained_dict_ = torch.load(pretrain_model_path, map_location=device)
# 加载模型时的键值要与保存时的对应
net_dict = net.state_dict()
pretrained_weight = {k: v for k,v in pretrained_dict_["net"].items() if k in net_dict}
net_dict.update(pretrained_weight)
net.load_state_dict(net_dict)
classifier_dict = classifier.state_dict()
pretrained_weight = {k: v for k,v in pretrained_dict_["classifier"].items() if k in classifier_dict}
classifier_dict.update(pretrained_weight)
classifier.load_state_dict(classifier_dict)
fusion_dict = fusion.state_dict()
pretrained_weight = {k: v for k,v in pretrained_dict_["fusion"].items() if k in fusion_dict}
fusion_dict.update(pretrained_weight)
fusion.load_state_dict(fusion_dict)
print("---model loaded---")



def test():
    y_pred_list = []
    y_true_list = []
    dt = [i/num_classes for i in range(num_classes)]
    # 生成一个csv文件，里面有预测值和真实值的对比
    mydict = []
    header = ["name", "y_pred", "y_true"]
    with torch.no_grad():
        for step, (patch, dist, y_true, imgname) in enumerate(test_dataloader):
            patch, dist, y_true = patch.type(torch.FloatTensor), dist.type(torch.FloatTensor), y_true.type(torch.FloatTensor)
            patch, dist, y_true = patch.to(device), dist.to(device), y_true.to(device)
            output = torch.zeros(len(y_true), out_features)
            output = output.to(device)
            patch = patch.float()
            # 把n_rater个预测结果拼接在一起（或相加），然后再通过classifier得到输出
            for t in range(n_rater):
                if use_fdmrm:
                    net.forward(patch[..., t], dist)
                    reconstruction = net.reconstruct(t)
                else:
                    reconstruction = net.forward(patch[..., t])
                if use_emf:
                    reconstruction = fusion(reconstruction)
                output = output + reconstruction
            predict_dist, y_pred = classifier(output)
            if weight[1] == 0:
                y_pred = y_pred
            else:
                y_pred = (weight[1]*dt[predict_dist.argmax(dim=-1)] + weight[2]*y_pred) / (weight[1]+weight[2])
            y_true = y_true.squeeze(0)
            y_true = y_true.cpu().data.numpy()
            y_true = y_true * num_classes
            y_pred = y_pred.squeeze(0)
            y_pred = y_pred.squeeze(0)
            y_pred = y_pred.cpu().data.numpy()
            y_pred = y_pred * num_classes
            y_pred_list.append(y_pred)
            y_true_list.append(y_true)
            mydict.append([imgname[0], y_pred, y_true])

        with open(os.path.join(result_path, "result.csv"), "w", newline='') as result:
            writer = csv.writer(result)
            writer.writerow(header)
            writer.writerows(mydict)
        y_true_list,y_pred_list = np.array(y_true_list),np.array(y_pred_list)
        abs_error = np.abs(y_true_list - y_pred_list)
        mae = np.mean(abs_error)
        print("MAE指标为：", mae)


if __name__ == "__main__":
    test()

