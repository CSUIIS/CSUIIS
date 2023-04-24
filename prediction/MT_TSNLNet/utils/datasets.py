#!coding:utf-8
import numpy as np
import torchvision as tv
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from prediction.MT_TSNLNet.utils.randAug import RandAugmentMC
from prediction.MT_TSNLNet.utils.data_utils import NO_LABEL
from prediction.MT_TSNLNet.utils.data_utils import TransformWeakStrong as wstwice

load = {}


def register_dataset(dataset):
    def warpper(f):
        load[dataset] = f
        return f

    return warpper


def encode_label(label):
    return NO_LABEL * (label + 1)


def decode_label(label):
    return NO_LABEL * label - 1


def split_relabel_data(np_labs, labels, label_per_class,
                       num_classes):
    """ Return the labeled indexes and unlabeled_indexes
    """
    labeled_idxs = []
    unlabed_idxs = []
    for id in range(num_classes):
        indexes = np.where(np_labs == id)[0]
        np.random.shuffle(indexes)
        labeled_idxs.extend(indexes[:label_per_class])
        unlabed_idxs.extend(indexes[label_per_class:])
    np.random.shuffle(labeled_idxs)
    np.random.shuffle(unlabed_idxs)
    ## relabel dataset
    for idx in unlabed_idxs:
        labels[idx] = encode_label(labels[idx])

    return labeled_idxs, unlabed_idxs


@register_dataset('cifar10')
def cifar10(n_labels, data_root='./data-local/cifar10/'):
    channel_stats = dict(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010])
    train_transform = transforms.Compose([
        transforms.Pad(2, padding_mode='reflect'),
        transforms.ColorJitter(brightness=0.4, contrast=0.4,
                               saturation=0.4, hue=0.1),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    trainset = tv.datasets.CIFAR10(data_root, train=True, download=True,
                                   transform=train_transform)
    evalset = tv.datasets.CIFAR10(data_root, train=False, download=True,
                                  transform=eval_transform)
    num_classes = 10
    label_per_class = n_labels // num_classes
    labeled_idxs, unlabed_idxs = split_relabel_data(
        np.array(trainset.train_labels),
        trainset.train_labels,
        label_per_class,
        num_classes)
    return {
        'trainset': trainset,
        'evalset': evalset,
        'label_idxs': labeled_idxs,
        'unlab_idxs': unlabed_idxs,
        'num_classes': num_classes
    }


@register_dataset('wscifar10')
def wscifar10(n_labels, data_root='./data-local/cifar10/'):
    channel_stats = dict(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010])
    weak = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Pad(2, padding_mode='reflect'),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    strong = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Pad(2, padding_mode='reflect'),
        transforms.RandomCrop(32),
        RandAugmentMC(n=2, m=10),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    train_transform = wstwice(weak, strong)
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    trainset = tv.datasets.CIFAR10(data_root, train=True, download=True,
                                   transform=train_transform)
    evalset = tv.datasets.CIFAR10(data_root, train=False, download=True,
                                  transform=eval_transform)
    num_classes = 10
    label_per_class = n_labels // num_classes
    labeled_idxs, unlabed_idxs = split_relabel_data(
        np.array(trainset.train_labels),
        trainset.train_labels,
        label_per_class,
        num_classes)
    return {
        'trainset': trainset,
        'evalset': evalset,
        'label_idxs': labeled_idxs,
        'unlab_idxs': unlabed_idxs,
        'num_classes': num_classes
    }


@register_dataset('cifar100')
def cifar100(n_labels, data_root='./data-local/cifar100/'):
    channel_stats = dict(mean=[0.5071, 0.4867, 0.4408],
                         std=[0.2675, 0.2565, 0.2761])
    train_transform = transforms.Compose([
        transforms.Pad(2, padding_mode='reflect'),
        transforms.ColorJitter(brightness=0.4, contrast=0.4,
                               saturation=0.4, hue=0.1),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    trainset = tv.datasets.CIFAR100(data_root, train=True, download=True,
                                    transform=train_transform)
    evalset = tv.datasets.CIFAR100(data_root, train=False, download=True,
                                   transform=eval_transform)
    num_classes = 100
    label_per_class = n_labels // num_classes
    labeled_idxs, unlabed_idxs = split_relabel_data(
        np.array(trainset.train_labels),
        trainset.train_labels,
        label_per_class,
        num_classes)
    return {
        'trainset': trainset,
        'evalset': evalset,
        'labeled_idxs': labeled_idxs,
        'unlabeled_idxs': unlabed_idxs,
        'num_classes': num_classes
    }


@register_dataset('mydataset')
def mydataset(n_labels, data_root='../../A2_data/202201/512/1月512有标签加无标签.csv'):
    channel_stats = dict(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
    train_transform = transforms.Compose([
        transforms.Pad(2, padding_mode='reflect'),
        # transforms.ColorJitter(brightness=0.4, contrast=0.4,
        #                        saturation=0.4, hue=0.1),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    # trainset = MyDataset('../../data202201/train_data.csv', r'../../data202201', train_transform)
    trainset = MyDataset('../../data202201/train_data.csv', r'../../data202201', train_transform)
    # evalset = MyDataset('../../data202201/val_data.csv', r'../../data202201', eval_transform)
    evalset = MyDataset('../../data202201/val_data.csv', r'../../data202201', eval_transform)
    num_classes = 5

    labeled_idxs = []
    unlabed_idxs = []
    for i in range(len(trainset)):
        if i % 100 == 0:
            print(i)
        if trainset[i][1] == -1:
            unlabed_idxs.append(i)
        else:
            labeled_idxs.append(i)
    # label_per_class = n_labels // num_classes
    # labeled_idxs, unlabed_idxs = split_relabel_data(
    #     np.array(trainset.train_labels),
    #     trainset.train_labels,
    #     label_per_class,
    #     num_classes)
    return {
        'trainset': trainset,
        'evalset': evalset,
        'label_idxs': labeled_idxs,
        'unlab_idxs': unlabed_idxs,
        'num_classes': num_classes
    }


import pandas as pd
from PIL import Image
import os
import torch


class MyDataset(Dataset):
    # Initialization
    def __init__(self, csv_path, data_path, transform):
        self.data_path = data_path
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.dirname = []
        self.label = np.zeros(len(self.df))
        for idx, row_data in self.df.iterrows():
            label = float(row_data['k+'])
            if label == -1:
                self.label[idx] = -1
            else:
                if label <= 20:
                    self.label[idx] = 0
                elif label <= 23:
                    self.label[idx] = 1
                elif label <= 26:
                    self.label[idx] = 2
                elif label <= 29:
                    self.label[idx] = 3
                else:
                    self.label[idx] = 4
            temp = ['{:02d}'.format(int(x)) for x in row_data['img_name'].split('_')[:3]]
            self.dirname.append((''.join(temp), row_data['img_name']))

    # Get item
    def __getitem__(self, index):
        if self.label[index] == -1:
            img_path = os.path.join('/home/mzl116/disk/Lisilong/FlotationProject/data202201/unlabeled_data',
                                    self.dirname[index][0], self.dirname[index][1])
        else:
            img_path = os.path.join('/home/mzl116/disk/Lisilong/FlotationProject/data202201/labeled_data',
                                    self.dirname[index][0], self.dirname[index][1])
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, torch.tensor(self.label[index],dtype=torch.long)

    # Get length
    def __len__(self):
        return len(self.df)
