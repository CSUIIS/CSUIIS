from torch.utils.data import Dataset
import torch
import os
import csv
from PIL import Image


# 从内存中读取数据
class MyDatasetMeo(Dataset):
    def __init__(self, imgs, train=True, transform=None):
        self.imgs = imgs
        self.transform = transform
        self.train = train

    def __getitem__(self,index):
        img, label = self.imgs[index]
        if self.transform is not None:
            img, label = self.transform(img, label)
        return img,label

    def __len__(self):
        return len(self.imgs)