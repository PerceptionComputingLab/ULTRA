import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["MKL_NUM_THREADS"] = '4'
os.environ["NUMEXPR_NUM_THREADS"] = '4'
os.environ["OMP_NUM_THREADS"] = '4'
import torchvision.transforms as transforms
import os
import glob
from PIL import Image
import csv
from model import get_backbone_model, MLP
from config import ConfigBackbone, ConfigLDL
from utils import *
from metrics import *
from tqdm import tqdm
import torch.nn as nn
import re
import argparse


def dict_sort(dict_):
    dict_ = sorted(dict_.items(), key=lambda x: x[0])
    list_out = [x[1] for x in dict_]
    return np.array(list_out)

def list_sort(list_in):
    list_out = sorted(list_in, key=lambda x: 1000*x[0]+x[1])
    return np.array(list_out)

def parse_string(config):
    '''
    parse a string of the directory generated from the training phase.
    '''
    # input_str = config.checkpoint_path
    input_str = os.path.basename(os.path.split(config.checkpoint_path)[0])
    re_style = r'[\_]+'
    split_text = re.split(re_style, input_str)
    if config.architecture == "LDL":
        model = split_text[1]
        std = float(split_text[2])
        num_discretizing = int(split_text[3])
        num_raters = int(split_text[4])
        architecture = split_text[5]

        config.model = model
        config.std = std
        config.num_discretizing = num_discretizing
        config.num_raters = num_raters
        config.architecture = architecture
    else:
        model = split_text[1]
        architecture = split_text[2]
        config.model = model
        config.architecture = architecture

    return config

def parse_string_new(input_str):
    '''
    parse a string of the directory generated from the training phase.
    '''
    re_style = r'[\_]+'
    split_text = re.split(re_style, input_str)
    
    model = split_text[1]
    std = float(split_text[2])
    num_discretizing = int(split_text[3])
    num_raters = int(split_text[4])
    architecture = split_text[5]
    return {"model":model, "std":std, "num_discretizing":num_discretizing, "num_raters":num_raters}

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('--checkpoint_path', type=str, help=' checkpoint path', default="runs/runs_resnet34_0.02_80_3_LDL_Jan18_01-10-32/best_valloss_metric.pt")
    parser.add_argument('--test_path', type=str, help='testing data path', default="/home/lixiangyu/Dataset/BreastPathQ/test_patches")
    args = parser.parse_args()
    return args


if __name__ == "__main__":


    args = parse_args()
    device = torch.device("cuda")
    # config = ConfigBackbone()
    config = ConfigLDL()
    config.checkpoint_path = args.checkpoint_path
    config.test_path = args.test_path

    config = parse_string(config)
    
    checkpoint = torch.load(config.checkpoint_path)
    if config.architecture == "LDL":
        backbone = get_backbone_model(config.model, as_backbone=True)
        mlp = MLP(input_dim=config.num_features, output_dim=config.num_discretizing)
        backbone.load_state_dict(checkpoint['backbone'])
        mlp.load_state_dict(checkpoint['mlp'])
        backbone.to(device)
        mlp.to(device)
        backbone.eval()
        mlp.eval()
    elif config.architecture== "backbone":
        backbone = get_backbone_model(config.model, as_backbone=False)
        # backbone.load_state_dict(checkpoint['backbone'])
        backbone.load_state_dict(checkpoint)
        backbone.to(device)
        backbone.eval()

    if config.phase == "validate":
        validation_path = os.path.join(config.data_path, "validation/*.tif")
        split = 5
         # Read the validation label set.
        labelpath = os.path.join(config.data_path, "val_labels.csv")
        with open(labelpath, mode='r') as infile:
            reader = csv.reader(infile)
            head = next(reader)
            mydict = {str(rows[0]) + "_" + str(rows[1]): float(rows[2]) for rows in reader}

    elif config.phase == "test":
        validation_path = os.path.join(config.test_path, "*.tif")
        split = 6
    else:
        validation_path = os.path.join(config.data_path, "train/*.tif")
        split= 5

    imgdir = glob.glob(validation_path)
    train_data_transforms = transforms.Compose([transforms.ToTensor()])
    predict = []
    label = []
    csvData = [['slide', 'rid', 'score']]
    with torch.no_grad():
        for img in tqdm(imgdir):
            imgname = os.path.basename(img)
            iname = os.path.splitext(imgname)[0]
            imag = Image.open(img)
            imag = train_data_transforms(imag)

            norm = transforms.Compose([transforms.Normalize([imag[0].mean(), imag[1].mean(), imag[2].mean()],
                                                        [imag[0].std(), imag[1].std(), imag[2].std()])])
            imag = norm(imag)
            imag = imag.unsqueeze(0)
            imag = imag.to(device)
            if config.phase != "test":
                label.append(mydict[iname])

            if config.architecture == "LDL":
                output = torch.empty(1, config.num_raters, config.num_features).cuda()
                for t in range(config.num_raters):
                    output[:, t, :] = backbone(imag)
                probs, pre = mlp(torch.mean(output, dim=1))
                out = compute_score(probs=probs, num_discretizing=config.num_discretizing)
                out_value = (out.item() + pre.item()) / 2
                # predict[iname] = out_value
                predict.append(out_value)
                
                csvData.append([iname[0:split], iname[split+1:], out_value])
            elif config.architecture == "backbone":
                out = backbone(imag)
                out = torch.sigmoid(out)
                predict.append(out.item())
                csvData.append([iname[0:split], iname[split+1:], out.item()])

    if config.phase != "test":
        # predict = dict_sort(predict)
        # label = dict_sort(label)
        icc_, _ = icc(predict,label)
        pk_ = pk(label, predict)
        kappa_, _ = kappa(predict, label)
        mse_, _ = mse(predict,label)

        print(f"icc{icc_}")
        print(f"pk{pk_}")
        print(f"kappa{kappa_}")
        print(f"mse{mse_}")
    else:
        pass

    # save to the csv file
    # file_path = os.path.dirname(config.checkpoint_path)[10:]
    file_path = os.path.basename(os.path.split(config.checkpoint_path)[0])
    with open(f'result/{file_path}_{config.phase}.csv', 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        # csv_new = list_sort(csvData)
        writer.writerows(csvData)
        if config.phase != "test":
            writer.writerow(["icc:"f'{icc_}', "pk:"f'{pk_}', "kappa:"f'{kappa_}', "mse:"f'{mse_}'])
    csvFile.close()
