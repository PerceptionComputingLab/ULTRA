from torch.utils.data.dataset import Dataset
import torch
from scipy import stats
import numpy as np
import cv2
import os
import glob
from tqdm import tqdm
from PIL import Image
import csv
import torchvision.transforms as transforms
import random
from utils import guassian_dist, tlgd_dist
from randaugment import *



class TranningDataSet(Dataset):
    def __init__(self, n_raters=3, sigma=0.2, dist_mode="guassian", num_classes=13, image=None, label=None, transform=False):
        self.n_raters = n_raters
        self.transform = transform
        self.dist_mode = dist_mode
        self.num_classes = num_classes
        self.imgdir = glob.glob(image)
        self.len = len(self.imgdir)
        self.augment_pool = augment_pool()
        self.m = 10
        self.n = 1
        with open(label, mode='r') as infile:
            reader = csv.reader(infile)
            self.mydict = {str(rows[0]) + "_" + str(rows[1]): rows[2] for rows in reader}
        del self.mydict['slide_rid']
        self.sigma = sigma

    def __getitem__(self, index):
        image = self.imgdir[index]
        imgname = os.path.basename(image)
        imgname = os.path.splitext(imgname)[0]
        image = Image.open(image)
        patch = np.zeros((3, 512, 512, self.n_raters))
        for t in range(self.n_raters):
            ops = random.choices(self.augment_pool, k=self.n)
            for op, minval, maxval in ops:
                v = np.random.randint(1, self.m)
                val = (float(v) / 30) * float(maxval - minval) + minval
                image = np.array(image)
                image = op(image, val)
                if isinstance(image, dict):
                    image = Image.fromarray(image['image'])
                    image = np.array(image)
                else:
                    image = image
                img_array = np.array(image).transpose(2, 0, 1)
                patch[:, :, :, t] = img_array
        label = float(self.mydict[str(imgname)])
        if self.dist_mode == "t_lgd":
            dist = tlgd_dist(label, self.sigma)
        elif self.dist_mode == "guassian":
            dist = guassian_dist(label, 0.01, self.num_classes)
        return patch, dist, label, imgname

    def __len__(self):
        return self.len

class TestingDataSet(Dataset):
    def __init__(self, n_raters=3, sigma=0.2, dist_mode="guassian", num_classes=13, image=None, label=None):
        self.n_raters = n_raters
        self.dist_mode = dist_mode
        self.num_classes = num_classes
        self.imgdir = glob.glob(image)
        self.len = len(self.imgdir)
        self.m = 10
        self.n = 1
        with open(label, mode='r') as infile:
            reader = csv.reader(infile)
            self.mydict = {str(rows[0]) + "_" + str(rows[1]): rows[2] for rows in reader}
        del self.mydict['slide_rid']
        self.sigma = sigma

    def __getitem__(self, index):
        image = self.imgdir[index]
        imgname = os.path.basename(image)
        imgname = os.path.splitext(imgname)[0]
        image = Image.open(image)
        patch = np.zeros((3, 512, 512, self.n_raters))
        for t in range(self.n_raters):
            image = np.array(image)
            img_array = np.array(image).transpose(2, 0, 1)
            patch[:, :, :, t] = img_array
        label = float(self.mydict[str(imgname)])
        if self.dist_mode == "t_lgd":
            dist = tlgd_dist(label, self.sigma)
        elif self.dist_mode == "guassian":
            dist = guassian_dist(label, 0.01, self.num_classes)
        return patch, dist, label, imgname

    def __len__(self):
        return self.len


class TrainBoneAge(Dataset):
    def __init__(self, n_raters=3, sigma=0.2, dist_mode="guassian", num_classes=13, image=None, label=None, transform=False, size=768):
        self.n_raters = n_raters
        self.transform = transform
        self.dist_mode = dist_mode
        self.num_classes = num_classes
        self.imgdir = image
        self.len = len(self.imgdir)
        self.augment_pool = augment_pool()
        self.m = 10
        self.n = 3
        self.label = label
        self.mydict = self.read_bone_age()
        self.sigma = sigma
        self.size = size
    
    def read_bone_age(self):
        with open(self.label, mode='r') as infile:
            reader = csv.reader(infile)
            mydict = {str(rows[0]): rows[1] for rows in reader}
        return mydict 

    def __getitem__(self, index):
        img_path = self.imgdir[index]
        imgname = os.path.basename(img_path)
        imgname = os.path.splitext(imgname)[0]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(self.size,self.size))
        image = np.asarray(image, dtype=np.uint8)
        patch = np.zeros((3, self.size, self.size, self.n_raters))
        # image = cv2.imread(img_path,0)
        # image = cv2.resize(image,(300,300))
        # image = np.asarray(image, dtype=np.uint8)
        # patch = np.zeros((1, 300, 300, self.n_raters))
        for t in range(self.n_raters):
            ops = random.choices(self.augment_pool, k=self.n)
            for op, minval, maxval in ops:
                v = np.random.randint(1, self.m)
                val = (float(v) / 30) * float(maxval - minval) + minval
                image = np.array(image)
                image = op(image, val)
                if isinstance(image, dict):
                    image = Image.fromarray(image['image'])
                    image = np.array(image)
                else:
                    image = image
                img_array = np.array(image).transpose(2, 0, 1)
                patch[:, :, :, t] = img_array
                # patch[0, :, :, t] = image
        label = float(self.mydict[str(imgname)])/250
        tmp = stats.norm.pdf(np.linspace(0, 1, self.num_classes), 
                             loc=label/self.num_classes, scale=1/self.num_classes).astype(np.float32)
        dist = tmp / tmp.sum()
        return patch, dist, label, imgname

    def __len__(self):
        return self.len
    

class TestBoneAge(Dataset):
    def __init__(self, n_raters=3, sigma=0.2, dist_mode="guassian", num_classes=13, image=None, label=None, size=768):
        self.n_raters = n_raters
        self.dist_mode = dist_mode
        self.num_classes = num_classes
        self.imgdir = image
        self.len = len(self.imgdir)
        self.m = 10
        self.n = 1
        self.label = label
        self.mydict = self.read_bone_age()
        self.sigma = sigma
        self.size = size
    
    def read_bone_age(self):
        with open(self.label, mode='r') as infile:
            reader = csv.reader(infile)
            mydict = {str(rows[0]): rows[2] for rows in reader}
        return mydict 

    def __getitem__(self, index):
        img_path = self.imgdir[index]
        imgname = os.path.basename(img_path)
        imgname = os.path.splitext(imgname)[0]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(self.size,self.size))
        image = np.asarray(image, dtype=np.uint8)
        patch = np.zeros((3, self.size, self.size, self.n_raters))
        # image = cv2.imread(img_path,0)
        # image = cv2.resize(image,(300,300))
        # image = np.asarray(image, dtype=np.uint8)
        # patch = np.zeros((1, 300, 300, self.n_raters))
        for t in range(self.n_raters):
            image = np.array(image)
            img_array = np.array(image).transpose(2, 0, 1)
            patch[:, :, :, t] = img_array
            # patch[0, :, :, t] = image
        label = float(self.mydict[str(imgname)])/250
        tmp = stats.norm.pdf(np.linspace(0, 1, self.num_classes), 
                             loc=label/self.num_classes, scale=1/self.num_classes).astype(np.float32)
        dist = tmp / tmp.sum()
        return patch, dist, label, imgname

    def __len__(self):
        return self.len
