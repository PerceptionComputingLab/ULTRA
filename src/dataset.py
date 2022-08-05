import os
os.environ["MKL_NUM_THREADS"] = '4'
os.environ["NUMEXPR_NUM_THREADS"] = '4'
os.environ["OMP_NUM_THREADS"] = '4'
from torch.utils.data.dataset import Dataset
import torch.utils.data as Data
import torch
import torchvision.transforms as transforms
import numpy as np
import os
import glob
from tqdm import tqdm
from PIL import Image
import csv
import torchvision.transforms as transforms
from utils import compute_target_dist
import random
           
to_tensor = transforms.Compose([transforms.ToTensor()])

def normalize(img):
    mean = np.mean(img, axis=(1,2))
    std = np.std(img, axis=(1,2))
    shape = img.shape
    norm = np.zeros(shape, dtype="float32")
    for i in range(shape[0]):
        norm[i] = (img[i]-mean[i])/std[i]
    return norm


class trainset(Dataset):
    '''
    data loader of BreatpathQ dataset
    '''
    def __init__(self, image=None, label=None, data_transform=None):
        with open(label, mode='r') as infile:
            reader = csv.reader(infile)
            mydict = {str(rows[0]) + "_" + str(rows[1]): rows[2] for rows in reader}
        del mydict['slide_rid']
        imgdir = glob.glob(image)
        self.len = len(imgdir)
        self.x_train = np.zeros((self.len, 3, 512, 512))
        self.y_train = np.zeros((self.len, 1))
        self.data_transform = data_transform
        n = 0
        for img in tqdm(imgdir):
            imgname = os.path.basename(img)
            iname = os.path.splitext(imgname)[0]
            imag = Image.open(img)
            if self.data_transform:
                imag = data_transform(imag)
                norm = transforms.Compose([transforms.Normalize([imag[0].mean(), imag[1].mean(), imag[2].mean()],
                                                                [imag[0].std(), imag[1].std(), imag[2].std()])])
                imag = norm(imag)
            else:
                imag = to_tensor(imag)
                norm = transforms.Compose([transforms.Normalize([imag[0].mean(), imag[1].mean(), imag[2].mean()],
                                                                [imag[0].std(), imag[1].std(), imag[2].std()])])
                imag = norm(imag)
            self.x_train[n] = imag
            self.y_train[n] = mydict[str(iname)]
            n += 1
        self.y_train = torch.from_numpy(self.y_train)

    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]

    def __len__(self):
        return self.len


class TranningDataSet(Dataset):
    def __init__(self, config, image=None,label=None,augment=False, n_raters=3):
        with open(label, mode='r') as infile:
            reader = csv.reader(infile)
            mydict = {str(rows[0]) + "_" + str(rows[1]): rows[2] for rows in reader}
        del mydict['slide_rid']
        self.augment = augment
        imgdir = glob.glob(image)
        self.len = len(imgdir)
        self.x_train = torch.zeros((self.len, 3, 512, 512, n_raters))
        self.y_train = np.zeros((self.len, config.num_discretizing))
        self.y_label = np.zeros((self.len))
        n = 0
        
        for img in tqdm(imgdir):
            imgname = os.path.basename(img)
            iname = os.path.splitext(imgname)[0]
            imag_orig = Image.open(img)
            for t in range(n_raters):
                if self.augment:
                    random.seed(t)
                    trans = transforms.Compose([
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomVerticalFlip(p=0.5)
                        ])
                    imag = trans(imag_orig)
                else:
                    imag = imag_orig

                # Normalize
                imag = to_tensor(imag)
                norm = transforms.Compose([transforms.Normalize([imag[0].mean(), imag[1].mean(), imag[2].mean()],
                                                                [imag[0].std(), imag[1].std(), imag[2].std()])])
                imag = norm(imag)
                self.x_train[n,:,:,:,t] = imag
            # compute label distribution
            y_value = float(mydict[str(iname)])
            self.y_label[n] = y_value
            y_temp = compute_target_dist(y_value, config.std, config.num_discretizing)
            self.y_train[n] = y_temp
            n += 1
        # self.x_train = torch.from_numpy(self.x_train)
        self.y_train = torch.from_numpy(self.y_train)
        self.y_label = torch.from_numpy(self.y_label)
    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index],self.y_label[index]

    def __len__(self):
        return self.len