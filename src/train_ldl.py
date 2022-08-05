import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["MKL_NUM_THREADS"] = '4'
os.environ["NUMEXPR_NUM_THREADS"] = '4'
os.environ["OMP_NUM_THREADS"] = '4'
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import torch.utils.data as Data
from logger import Logger
from config import ConfigLDL
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm
import argparse
from utils import *
from dataset import trainset, TranningDataSet
from model import get_backbone_model, MLP
from metrics import *
import time


def parse_args():
    '''
    Parsing important arguments in CMD
    '''
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('--model', type=str, help='train model', default="resnet34")
    parser.add_argument('--std', type=float, help='hyper-parameter of the std for gaussian distribution', default=0.02)
    parser.add_argument('--num_discretizing', type=int, help='number of samplings for gaussian distribution', default=100)
    parser.add_argument('--num_raters', type=int, help='number of raters', default=3)
    parser.add_argument('--lr_S', type=float, help='learning_rate', default=1e-4)
    parser.add_argument('--batchsize', type=int, help='batch size', default=8)
    parser.add_argument('--pre_train_path', type=str, help='pretrain model path', default='pretrained/pretrain_model.pt')
    parser.add_argument('--data_path', type=str, help='training and validation data path', default="/home/lixiangyu/Dataset/BreastPathQ")
    args = parser.parse_args()
    return args


def iteration(dataloader, config, criterion, optimizer, backbone, mlp, mse_loss, mode="train"):
    '''
    iteration in trianing and validation phase.
    '''
    if mode == "train":
        backbone.train()
        mlp.train()
    else:
        backbone.eval()
        mlp.eval()
        predict_list = []
        label_list = []

    loss_epoch = 0
    loss_mse = 0
    for i, batch in enumerate(dataloader):
        inputs, targets, label = batch
        inputs, targets, label = inputs.type(torch.FloatTensor), targets.type(torch.FloatTensor), label.type(torch.FloatTensor)
        inputs, targets, label = inputs.cuda(), targets.cuda(), label.cuda()

        output = torch.empty(len(label), config.num_raters, config.num_features).cuda()
        for t in range(config.num_raters):
            output[:, t, :] = backbone(inputs[..., t])
        probs, pre = mlp(torch.mean(output, dim=1))
        out = compute_score(probs=probs, num_discretizing=config.num_discretizing)

        loss1 = compute_loss(criterion=criterion, probs=probs, target=targets)
        loss2 = mse_loss(pre.squeeze(1), label)
        # Test only loss1
        loss = loss1 + 20*loss2
        # loss = loss1
        loss_epoch += loss.item()
        
        # loss_mse += loss2.item()
        if mode == "train":
            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            label = label.squeeze(0)
            out = out.squeeze(0)
            pre = pre.squeeze()
            pre_numpy = pre.cpu().data.numpy()
            out_numpy = out.cpu().data.numpy()
            average_out = (pre_numpy+out_numpy)/2

            predict_list.append(average_out)
            label_list.append(label.cpu().data.numpy())

    if mode == "train":
        return loss_epoch, loss_mse
    else:
        icc_, _ = icc(predict_list,label_list)
        kappa_, _ = kappa(predict_list, label_list)
        mse_, _ = mse(predict_list,label_list)
        pk_ = pk(label_list, predict_list)
        return loss_epoch, loss_mse, icc_, kappa_, mse_, pk_


if __name__ == '__main__':

    args = parse_args()
    train_model = args.model
    std = args.std
    best_score = float('inf')
    best_metric = 0
    best_pk = float('inf')
    config = ConfigLDL()
    config.model = train_model
    config.std = std
    config.num_discretizing=args.num_discretizing
    config.num_raters = args.num_raters
    config.lr_S = args.lr_S
    config.batchsize = args.batchsize
    config.data_path = args.data_path

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_path = f"runs_{config.model}_{config.std}_{config.num_discretizing}_{config.num_raters}_{config.architecture}_{current_time}"
    log = Logger(os.path.join("runs", log_path), write=config.write_log, save_freq=4)
    log.log(config.get_str_config())

    # construct the basic model
    backbone = get_backbone_model(train_model, config.pretrained, 
                                        num_features=config.num_features, as_backbone=True)
    backbone = backbone.cuda()
    mlp = MLP(input_dim=config.num_features, output_dim=config.num_discretizing).cuda()
    pretrained_dict = torch.load(args.pre_train_path)
    backbone_dict = backbone.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in backbone_dict}
    # 2. overwrite entries in the existing state dict
    backbone_dict.update(pretrained_dict)
    # 3. load the new state dict
    backbone.load_state_dict(backbone_dict)
    #for p in backbone.parameters():
    #    p.requires_grad = False
    #for p in backbone.fc.parameters():
    #    p.requires_grad = True
    #for name, value in backbone.named_parameters():
    #    print('name: {},\t grad: {}'.format(name, value.requires_grad))
    #backbone = nn.DataParallel(backbone)
    #mlp = nn.DataParallel(mlp)
    # criterion_S = nn.MSELoss().cuda()
    criterion_S = nn.KLDivLoss(reduction='batchmean').cuda()
    mse_Loss = nn.MSELoss().cuda()
    optimizer_S = optim.Adam(list(backbone.parameters()) + list(mlp.parameters()), lr=config.lr_S, weight_decay=1e-5,
                             betas=(0.97, 0.999))
    # scheduler_S = optim.lr_scheduler.ReduceLROnPlateau(optimizer_S, factor=0.3, min_lr=1e-7)
    scheduler_S = optim.lr_scheduler.StepLR(optimizer_S, step_size=config.lr_step_size, gamma=config.lr_ratio)
    train_set = TranningDataSet(config, image=os.path.join(config.data_path, "train/*.tif"),
                                label=os.path.join(config.data_path, 'train_labels.csv'), augment=config.augmentation, n_raters=config.num_raters)
    val_set = TranningDataSet(config, image=os.path.join(config.data_path, "validation/*.tif"),
                              label=os.path.join(config.data_path, 'val_labels.csv'), augment=config.augmentation, n_raters=config.num_raters)
    train_dataloader = Data.DataLoader(train_set, batch_size=config.batchsize, shuffle=True, pin_memory=True,
                                       drop_last=True)
    val_dataloader = Data.DataLoader(val_set, batch_size=1, shuffle=False, pin_memory=True)
    log.log("start training!")
    log.log('%7s | %7s | %7s | %7s | %7s | %0.7s |%0.7s | %0.7s | %0.7s | %0.7s |%0.7s |' % (
            "rate", "epoch", "Loss_train", "Loss_val", "MSE_train", "MSE_val", "icc", "kappa", "MSE", "pk", "Elapsed"))
    for epoch in range(config.num_epoch):
        start_epoch_time = time.time()
        loss_epoch_train, train_mse = iteration(train_dataloader, config, criterion_S, optimizer_S, backbone, mlp, mse_Loss, "train")
        loss_epoch_train, train_mse = loss_epoch_train / train_set.__len__(), train_mse / train_set.__len__()

        # online evaluation
        with torch.no_grad():
            loss_epoch_val, val_mse, icc_, kappa_, mse_, pk_ = iteration(val_dataloader, config, criterion_S, optimizer_S, backbone, mlp, mse_Loss, "validate")
            loss_epoch_val, val_mse = loss_epoch_val / val_set.__len__(), val_mse / val_set.__len__()

        end_epoch_time = time.time()
        elapsed = end_epoch_time-start_epoch_time
        log.log('%0.6f | %6d | %0.5f | %0.5f | %0.5f | %0.5f |%0.5f | %0.5f | %0.5f | %0.5f |%0.5f |' % (
            optimizer_S.state_dict()["param_groups"][0]['lr'], epoch, loss_epoch_train, loss_epoch_val, train_mse, val_mse,
            icc_, kappa_, mse_, pk_, elapsed))

        log.write_to_board(f"Loss", {"train_loss": loss_epoch_train, "val_loss": loss_epoch_val}, epoch)
        log.write_to_board(f"Metric", {"ICC": icc_, "kappa":kappa_, "mse":mse_, "pk":pk_}, epoch)
        
        if epoch % 5 == 0:
            log.save_model({'epoch': epoch,
                            'backbone': backbone.state_dict(),
                            'mlp': mlp.state_dict()},
                           f"checkpoint_{epoch}.pt", forced=True)

        total_metric = icc_+kappa_-mse_

        if total_metric > best_metric:
            log.save_model({'epoch': epoch,
                            'backbone': backbone.state_dict(),
                            'mlp': mlp.state_dict(),
                            'best_score': total_metric},
                           "best_valloss_metric.pt", forced=True)
            log.save_model({'epoch': epoch,
                            'backbone': backbone.state_dict(),
                            'mlp': mlp.state_dict(),
                            'best_score': total_metric},
                           f"best_valloss_{epoch}_{loss_epoch_val}.pt", forced=True)
            best_metric = total_metric
        
        

        # learning rate decay
        scheduler_S.step()

