
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
import shutil
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
    parser = argparse.ArgumentParser(description='Train SLDL')
    parser.add_argument('--experiment', type=str, help='tag', default='experiment')
    parser.add_argument('--backbone', type=str, help='backbone', default='resnet34')
    parser.add_argument('--gpu', type=int, help='gpu序号', default=0)
    parser.add_argument('--dist_mode', type=str, help='标签生成是用guassian还是tlgd', default="guassian")
    parser.add_argument('--use_emf', type=str, help='是否用sigmoid自监督', default="True")
    parser.add_argument('--use_fdmrm', type=str, help='是否用vae', default="True")
    parser.add_argument('--n_rater', type=int, help='分支数目', default=1)
    parser.add_argument('--sigma', type=float, help='温度系数', default=0.1)
    parser.add_argument('--weight', nargs='+', help='不同loss权重', default=[0.1, 0.1, 10, 0])
    parser.add_argument('--data_path', type=str, help='数据路径', default="/home/lxj/bone_age")
    parser.add_argument('--log_path', type=str, help='训练日志路径', default="./log")
    parser.add_argument('--checkpoints_path', type=str, help='模型保存路径', default="./checkpoints")
    parser.add_argument('--result_path', type=str, help='预测结果保存路径', default="./result")
    parser.add_argument('--pretrain_model_path', type=str, help='预训练模型路径', default=None)
    args = parser.parse_args()
    return args


def val():
    y_pred_list = []
    y_true_list = []
    dt = [i/num_classes for i in range(num_classes)]
    mydict = []
    header = ["name", "y_pred", "y_true"]
    with torch.no_grad():
        for step, (patch, dist, y_true, imgname) in enumerate(val_dataloader):
            patch, dist, y_true = patch.type(torch.FloatTensor), dist.type(torch.FloatTensor), y_true.type(torch.FloatTensor)
            patch, dist, y_true = patch.to(device), dist.to(device), y_true.to(device)
            output = torch.zeros(len(y_true), out_features)
            output = output.to(device)
            patch = patch.float()
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

        y_true_list,y_pred_list = np.array(y_true_list),np.array(y_pred_list)
        abs_error = np.abs(y_true_list - y_pred_list)
        mae = np.mean(abs_error)
        print("MAE指标为：", mae)
        global best_mae
        if mae < best_mae:
            best_mae = mae
            torch.save({'mae_': mae, 'net': net.state_dict(), 'classifier': classifier.state_dict(),'fusion': fusion.state_dict()},
                    os.path.join(checkpoints_path, f"mae{mae}.pt"))
            with open(os.path.join(result_path, "best_mae{}.csv".format(best_mae)), "w", newline='') as result:
                writer = csv.writer(result)
                writer.writerow(header)
                writer.writerows(mydict)
        return mae



def train():
    for epoch in range(epochs):
        loss_epoch = [0, 0, 0, 0, 0, 0]
        for step, (patch, dist, label, imgname) in enumerate(train_dataloader):
            patch, dist, label = patch.type(torch.FloatTensor), dist.type(torch.FloatTensor), label.type(torch.FloatTensor)
            patch, dist, label = patch.to(device), dist.to(device), label.to(device)
            vae_loss = torch.tensor(0).type(torch.FloatTensor).to(device)
            output = torch.zeros(len(label), out_features)
            output = output.to(device)
            patch = patch.float()
            for t in range(n_rater):

                if use_fdmrm:
                    net.forward(patch[..., t], dist)
                    fdmrm, reconstruction = net.fdmrm(t)
                    reg_loss = l2_regularisation(net.esfg_posterior) + l2_regularisation(net.esfg_prior) + \
                        l2_regularisation(net.eifg) + l2_regularisation(net.fcomb.mlp)
                    vae_loss += fdmrm + 1e-6 * reg_loss
                else:
                    reconstruction = net.forward(patch[..., t])

                if use_emf:
                    reconstruction = fusion(reconstruction)
                output = output + reconstruction

            vae_loss = vae_loss / n_rater
            predict_dist, predict_label = classifier(output)
            index = torch.argmax(dist, dim=1).to(device)
            kl_loss = KL(torch.log(predict_dist + 1e-7), dist)
            predict_label = predict_label.squeeze(1)
            mse_loss = MSE(predict_label, label)
            mae_loss = MAE(predict_label, label)
            uni_loss = unimodal_loss(predict_dist, index)
            total_loss = weight[0] * vae_loss + weight[1] * kl_loss + weight[2] * mse_loss + weight[3] * uni_loss + 10 * mae_loss
            loss_epoch[0] += total_loss.item()
            loss_epoch[1] += vae_loss.item()
            loss_epoch[2] += kl_loss.item()
            loss_epoch[3] += mse_loss.item()
            loss_epoch[4] += uni_loss.item()
            loss_epoch[5] += mae_loss.item()
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            #scheduler.step()
        mae= val()
        step = train_set.__len__()/batch_size
        writer.add_scalars("time", {"total_loss": loss_epoch[0] / step, "vae_loss": loss_epoch[1] / step,
                                    "kl_loss": loss_epoch[2] / step, "mse_loss": loss_epoch[3] / step,
                                    "uni_loss": loss_epoch[4] / step, "mae_loss": loss_epoch[5] / step,
                                    "mae" : mae}, epoch)

        with open(os.path.join(log_path, "logs.txt"), "a") as logs:
            logs.write("{:<10.6f}{:<10d}{:<10.5f}{:<10.5f}{:<10.5f}{:<10.5f}{:<10.5f}{:<10.5f}\n".format(
                    optimizer.state_dict()["param_groups"][0]['lr'], epoch, loss_epoch[0] / step,
                    loss_epoch[1] / step, loss_epoch[2] /step, loss_epoch[3] / step, loss_epoch[4] / step, mae))
        if epoch % 50 == 49:
            with open(os.path.join(log_path, "logs.txt"), "a") as logs:
                logs.write("{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}\n".format(
                        "rate", "epoch", "total", "vae", "kl", "mse", "unimodal", "mae"))


if __name__ == "__main__":
    try:
        args = parse_args()
        experiment = str(args.experiment)
        backbone = str(args.backbone)
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
        log_path = args.log_path
        checkpoints_path = args.checkpoints_path
        result_path = args.result_path
        pretrain_model_path = args.pretrain_model_path
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        log_path = os.path.join(log_path, "{}_".format(experiment) + current_time)
        checkpoints_path = os.path.join(checkpoints_path,"{}_".format(experiment) + current_time)
        result_path = os.path.join(result_path, "{}_".format(experiment) + current_time)
        os.makedirs(log_path)
        os.makedirs(result_path)
        os.makedirs(checkpoints_path)
        writer = SummaryWriter(log_dir=log_path)

        # 使用tlgd生成的标签分布的维度
        num_classes = 13  
        # 使用guassian生成的标签分布的维度
        if dist_mode == "guassian":  
            num_classes = 250
        # eifg生成的特征向量的维度
        num_filters = 128  
        # esfg生成的隐空间的特征向量的维度
        latent_dim = 6  
        out_features = num_classes
        epochs = 200
        batch_size = 8

        train_image = glob.glob(
            os.path.join(data_path, "preprocessed/boneage-training-dataset/*.png"))
        val_image = glob.glob(
            os.path.join(data_path, "preprocessed/boneage-validation-dataset/*.png"))
        train_set = TrainBoneAge(n_raters=n_rater, sigma=sigma, dist_mode=dist_mode,
                                    num_classes=num_classes, image=train_image,
                                    label=os.path.join(data_path, 'boneage-training-dataset.csv'))
        train_dataloader = Data.DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)

        val_set = TestBoneAge(n_raters=n_rater, sigma=sigma, dist_mode=dist_mode,
                                num_classes=num_classes, image=val_image,
                                label=os.path.join(data_path, 'boneage-validation-dataset.csv'))
        val_dataloader = Data.DataLoader(val_set, batch_size=1, shuffle=False, pin_memory=True)
        print("---data loaded---")

        net = Baseline(out_features, backbone).to(device)
        if use_fdmrm:
            net = ULTRA(in_features=256, out_features=out_features, num_classes=num_classes, num_filters=num_filters, latent_dim=latent_dim,
                        n_raters=n_rater, training=True, device=device, backbone=backbone)
       
        classifier = Classifier(in_features=out_features , num_classes=num_classes).to(device)
        fusion = Fusion().to(device)

        optimizer = Adam([*net.parameters()] + [*classifier.parameters()], lr=1e-4, weight_decay=1e-4,betas=(0.9, 0.999))
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
        KL = nn.KLDivLoss(reduction='batchmean').to(device)
        MSE = nn.MSELoss().to(device)
        MAE = nn.L1Loss().to(device)

        n = 0
        best_mae = 999

        if pretrain_model_path is not None:
            pretrained_dict_ = torch.load(pretrain_model_path, map_location=device)
            net_dict = net.state_dict()
            pretrained_weight = {k: v for k,v in pretrained_dict_["net"].items() if k in net_dict}
            net_dict.update(pretrained_weight)
            net.load_state_dict(net_dict)
            # classifier_dict = classifier.state_dict()
            # pretrained_weight = {k: v for k,v in pretrained_dict_["classifier"].items() if k in classifier_dict}
            # classifier_dict.update(pretrained_weight)
            # classifier.load_state_dict(classifier_dict)
            # fusion_dict = fusion.state_dict()
            # pretrained_weight = {k: v for k,v in pretrained_dict_["fusion"].items() if k in fusion_dict}
            # fusion_dict.update(pretrained_weight)
            # fusion.load_state_dict(fusion_dict)
            print("---model loaded---")

        print("---start training---")
        with open(os.path.join(log_path, "logs.txt"), "a") as logs:
            logs.write("dist_mode:{}\tuse_emf:{}\tuse_fdmrm:{}\tn_rater:{}\tsigma:{}\tweight:{}\n".format(
                dist_mode, use_emf, use_fdmrm, n_rater, sigma, weight))
            logs.write("{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}\n".format(
                "rate", "epoch", "total", "vae", "kl", "mse", "unimodal", "mae"))
        train()
    except:
        shutil.rmtree(log_path)
        shutil.rmtree(result_path)
        shutil.rmtree(checkpoints_path)

