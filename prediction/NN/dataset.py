from torch.utils.data import Dataset
import torch
from torch.utils.data.sampler import Sampler
import os
import itertools
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


class MyDataset_NN(Dataset):

    # Initialization
    def __init__(self, data, label, mode='2D'):
        self.data, self.label, self.mode = data, label, mode

    # Get item
    def __getitem__(self, index):
        if self.mode == '2D':
            return self.data[index, :], self.label[index, :]
        elif self.mode == '3D':
            return self.data[:, index, :], self.label[:, index, :]

    # Get length
    def __len__(self):
        if self.mode == '2D':
            return self.data.shape[0]
        elif self.mode == '3D':
            return self.data.shape[1]

# class MyDataset(Dataset):
#     # Initialization
#     def __init__(self, data, label, mode='2D'):
#         self.data, self.label, self.mode = data, label, mode
#         self.transform = None
#         self.num_classes = 5
#
#     # Get item
#     def __getitem__(self, index):
#         if self.mode == '2D':
#             return self.data[index, :], self.label[index, :] ,torch.LongTensor(index)
#         elif self.mode == '3D':
#             return self.data[:, index, :], self.label[:, index, :]
#
#     # Get length
#     def __len__(self):
#         if self.mode == '2D':
#             return self.data.shape[0]
#         elif self.mode == '3D':
#             return self.data.shape[1]

# MeanTeacher
class MyDataset_v2(Dataset):
    # Initialization
    def __init__(self, data, data_2, label, mode='2D'):
        self.data, self.data_2, self.label, self.mode = data, data_2, label, mode
        self.transform = None
        self.num_classes = 5

    # Get item
    def __getitem__(self, index):
        if self.mode == '2D':
            return (self.data[index, :], self.data_2[index, :]), self.label[index, :]
        elif self.mode == '3D':
            return self.data[:, index, :], self.label[:, index, :]

    # Get length
    def __len__(self):
        if self.mode == '2D':
            return self.data.shape[0]
        elif self.mode == '3D':
            return self.data.shape[1]


# MeanTeacher
class MyDataset_v1(Dataset):

    # Initialization
    def __init__(self, data, label, mode='2D'):
        self.data, self.label, self.mode = data, label, mode
        self.transform = None
        self.num_classes = 5

    # Get item
    def __getitem__(self, index):
        if self.mode == '2D':
            return (self.data[index, :], self.data[index, :]), self.label[index, :]
        elif self.mode == '3D':
            return self.data[:, index, :], self.label[:, index, :]

    # Get length
    def __len__(self):
        if self.mode == '2D':
            return self.data.shape[0]
        elif self.mode == '3D':
            return self.data.shape[1]


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.primary_batch_size = batch_size - secondary_batch_size
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            secondary_batch + primary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    args = [iter(iterable)]*n
    return zip(*args)