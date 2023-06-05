import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["MKL_NUM_THREADS"] = '4'
os.environ["NUMEXPR_NUM_THREADS"] = '4'
os.environ["OMP_NUM_THREADS"] = '4'
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import torch.utils.data as Data
from logger import Logger
from config import ConfigBackbone
import numpy as np
import os
from tqdm import tqdm
import argparse
from utils import *
from dataset import trainset, TranningDataSet
from model import get_backbone_model, MLP
from metrics import *
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('--model', type=str, help='train model', default="resnet34")
    parser.add_argument('--lr_S', type=float, help='learning_rate', default=1e-4)
    parser.add_argument('--batchsize', type=int, help='batch size', default=8)
    parser.add_argument('--data_path', type=str, help='training and validation data path', default="/home/lixiangyu/Dataset/BreastPathQ")
    args = parser.parse_args()
    return args


def iteration(dataloader, criterion, optimizer, backbone, mode="train"):
    '''
    iteration in trianing and validation phase.
    '''
    if mode == "train":
        backbone.train()
    else:
        backbone.eval()
        predict_list = []
        label_list = []

    loss_epoch = 0
    for i, batch in enumerate(dataloader):
        inputs, targets = batch
        inputs, targets = inputs.type(torch.FloatTensor), targets.type(torch.FloatTensor)
        inputs, targets = inputs.cuda(), targets.cuda()

        pre = backbone(inputs)
        output = torch.sigmoid(pre)
        loss = criterion(output, targets)
        loss_epoch += loss.item()
        
        if mode == "train":
            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            output = output.squeeze()
            targets = targets.squeeze()
            predict_list.append(output.cpu().data.numpy())
            label_list.append(targets.cpu().data.numpy())

    if mode == "train":
        return loss_epoch
    else:
        icc_, _ = icc(predict_list,label_list)
        kappa_ = kappa(predict_list, label_list)
        # mse_ = mse(predict_list,label_list)
        pk_ = pk(label_list, predict_list)
        return loss_epoch, icc_, kappa_, pk_


if __name__ == '__main__':
    args = parse_args()
    train_model = args.model
    best_score = float('inf')
    best_pk = float('inf')
    best_metric = 0
    config = ConfigBackbone()
    config.model = train_model
    config.lr_S = args.lr_S
    config.batchsize = args.batchsize
    config.data_path = args.data_path

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_path = f"runs_{config.model}_{config.architecture}_{current_time}"

    log = Logger(os.path.join("runs", log_path), write=True, save_freq=4)
    log.log(config.get_str_config())

    # construct the basic model
    backbone = get_backbone_model(train_model, config.pretrained, config.num_features, as_backbone=False)
    backbone = backbone.cuda()
    criterion_S = nn.MSELoss().cuda()
    optimizer_S = optim.Adam(backbone.parameters(), lr=config.lr_S, weight_decay=1e-5,
                             betas=(0.97, 0.999))
    # scheduler_S = optim.lr_scheduler.ReduceLROnPlateau(optimizer_S, factor=0.3, min_lr=1e-7)
    scheduler_S = optim.lr_scheduler.StepLR(optimizer_S, step_size=config.lr_step_size, gamma=config.lr_ratio)
    train_set = trainset(image=os.path.join(config.data_path, "train/*.tif"),
                                label=os.path.join(config.data_path, 'train_labels.csv'))
    val_set = trainset(image=os.path.join(config.data_path, "validation/*.tif"),
                              label=os.path.join(config.data_path, 'val_labels.csv'))
    train_dataloader = Data.DataLoader(train_set, batch_size=config.batchsize, shuffle=True, pin_memory=True,
                                       drop_last=True)
    val_dataloader = Data.DataLoader(val_set, batch_size=1, shuffle=False, pin_memory=True)
    log.log("start training!")
    log.log('   Rate   |  epoch  | Loss_train |  Loss_val  |icc | kappa |  pk ')
    for epoch in range(config.num_epoch):
        loss_epoch_train = iteration(train_dataloader, criterion_S, optimizer_S, backbone, "train")
        loss_epoch_train = loss_epoch_train / train_set.__len__()

        # online evaluation
        with torch.no_grad():
            loss_epoch_val, icc_, kappa_, pk_ = iteration(val_dataloader, criterion_S, optimizer_S, backbone, "validate")
            loss_epoch_val = loss_epoch_val / val_set.__len__()
    
        log.log('%0.6f | %6d | %0.5f | %0.5f | %0.5f | %0.5f |%0.5f |' % (
            optimizer_S.state_dict()["param_groups"][0]['lr'], epoch, loss_epoch_train, loss_epoch_val,
            icc_, kappa_, pk_))

        log.write_to_board(f"Loss", {"train_loss": loss_epoch_train, "val_loss": loss_epoch_val}, epoch)
        log.write_to_board(f"Metric", {"ICC": icc_, "kappa":kappa_, "pk": pk_}, epoch)
        
        total_metric = icc_ + kappa_ - loss_epoch_val
        # if epoch % 5 == 0:
        #     log.save_model({'epoch': epoch,
        #                     'backbone': backbone.state_dict(),
        #                     'mlp': mlp.state_dict()},
        #                    f"checkpoint_{epoch}.pt", forced=True)

        # save best checkpoint
        if loss_epoch_val < best_score:
            log.save_model({'epoch': epoch,
                            'backbone': backbone.state_dict(),
                            'best_score': best_score},
                           "best_valloss.pt", forced=True)
            # log.save_model({'epoch': epoch,
            #                 'backbone': backbone.state_dict(),
            #                 'best_score': best_score},
            #                f"best_valloss_{epoch}_{loss_epoch_val}.pt", forced=True)
            best_score = loss_epoch_val

        if total_metric > best_metric:
            log.save_model({'epoch': epoch,
                            'backbone': backbone.state_dict(),
                            'best_score': total_metric},
                           "best_valloss_metric.pt", forced=True)
            log.save_model({'epoch': epoch,
                            'backbone': backbone.state_dict(),
                            'best_score': total_metric},
                           f"best_valloss_{epoch}_{loss_epoch_val}.pt", forced=True)
            best_metric = total_metric
        # learning rate decay
        scheduler_S.step()
