from torch.utils import data
import os
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToTensor
import numpy
import torch
import glob
import random
import numpy as np
import pickle
from data import util

def augment(lr, hr, p,  hflip=True, rot=True):


    if random.random() > 0.5 and hflip:
        lr = lr[:, ::-1, :]
        hr = hr[:, ::-1, :]
        p = p[:, ::-1, :]
     

    if rot:
        rot_rand = random.random()
        if rot_rand > 0.75:
            lr = np.rot90(lr, k=1, axes=(0, 1))
            hr = np.rot90(hr, k=1, axes=(0, 1))
            p = np.rot90(p, k=1, axes=(0, 1))
        elif rot_rand > 0.5:
            lr = np.rot90(lr, k=2, axes=(0, 1))
            hr = np.rot90(hr, k=2, axes=(0, 1))
            p = np.rot90(p, k=2, axes=(0, 1))

        elif rot_rand > 0.25:
            lr = np.rot90(lr, k=3, axes=(0, 1))
            hr = np.rot90(hr, k=3, axes=(0, 1))
            p = np.rot90(p, k=3, axes=(0, 1))

    return lr, hr, p
class Data(data.Dataset):
    def __init__(self, root, args, train=False):

        self.args = args
        self.imgs_HR_path = os.path.join(root, 'HR')
        self.imgs_HR = sorted(
            glob.glob(os.path.join(self.imgs_HR_path, '*.png'))
        )
        
        if self.args.scale == 8:
            self.imgs_LR_path = os.path.join(root, 'LR_bicubic')
        elif self.args.scale == 16:
            self.imgs_LR_path = os.path.join(root, 'LR_x16_bicubic')
        elif self.args.scale == 4:
            self.imgs_LR_path = os.path.join(root, 'LR_x4_bicubic')
        self.imgs_LR = sorted(
            glob.glob(os.path.join(self.imgs_LR_path, '*.png'))
        )

        self.imgs_parsing_path = os.path.join(root, 'global_2')


        self.imgs_parsing = sorted(
            glob.glob(os.path.join(self.imgs_parsing_path, '*.png'))
        )
    
        self.transform = transforms.ToTensor()
        self.train = train

    def __getitem__(self, item):

        img_path_LR = os.path.join(self.imgs_LR_path, self.imgs_LR[item])
        img_path_HR = os.path.join(self.imgs_HR_path, self.imgs_HR[item])
        img_path_parsing = os.path.join(self.imgs_parsing_path, self.imgs_parsing[item])
        
        LR = Image.open(img_path_LR)
        HR = Image.open(img_path_HR)
        parsing = Image.open(img_path_parsing)
        # img_es_parsing = Image.open(img_es_parsing)
        HR = numpy.array(HR)
        LR = numpy.array(LR)
        parsing = numpy.array(parsing)
        if self.args.augment and self.train:
            LR, HR, parsing = augment(LR, HR, parsing)
       
        LR = np.ascontiguousarray(LR)
        HR = np.ascontiguousarray(HR)
        parsing = np.ascontiguousarray(parsing)
        HR = ToTensor()(HR)
        LR = ToTensor()(LR)
        res = ToTensor()(parsing)
        filename = os.path.basename(img_path_HR)

        return LR, HR, res, filename


    def __len__(self):
        return len(self.imgs_HR)


