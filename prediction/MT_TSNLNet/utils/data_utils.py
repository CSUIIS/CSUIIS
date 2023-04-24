import os
import itertools
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

NO_LABEL = -1
np.random.seed(1024)


class DataSetWarpper(Dataset):
    """Enable dataset to output index of sample
    """

    def __init__(self, dataset, num_classes):
        self.dataset = dataset
        self.num_classes = num_classes

    def __getitem__(self, index):
        sample, label = self.dataset[index]
        return sample, label, index

    def __len__(self):
        return len(self.dataset)


class TransformTwice:

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


class TransformWeakStrong:

    def __init__(self, trans1, trans2):
        self.transform1 = trans1
        self.transform2 = trans2

    def __call__(self, inp):
        out1 = self.transform1(inp)
        out2 = self.transform2(inp)
        return out1, out2


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.primary_batch_size = batch_size - secondary_batch_size
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size

        # 数据长度应大于batch size
        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    # 每次调用会返回一个batch的索引，指导所有无标签数据的索引返回完毕，而有标签数据可能返回多次索引
    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            secondary_batch + primary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


# 对传入序列进行随机排序
def iterate_once(iterable):
    return np.random.permutation(iterable)


# 以输入[1,2,3,4,5]为例，返回为[1,2,3,4,5,  2,4,5,3,1,  2,3,4,5,1 ,...]的迭代器
def iterate_eternally(indices, is_shuffle=True):
    shuffleFunc = np.random.permutation if is_shuffle else lambda x: x
    # 每次调用返回一个迭代器，其中包含所有传入的indices，是否打乱取决于is_shuffle
    def infinite_shuffles():
        while True:
            yield shuffleFunc(indices)
    # itertools.chain.from_iterable(迭代器) 迭代器内每个元素也是可迭代的，把所有子迭代器的元素放入同一个迭代器
    return itertools.chain.from_iterable(infinite_shuffles())


# 输入[1,2,3,4,5] n=2     返回[(0,1),(2,3)]
def grouper(iterable, n):
    # 每次调用迭代器会指向下一个
    args = [iter(iterable)] * n
    return zip(*args)


def test(primary_iter, secondary_iter):
    primary_iter = iterate_once(primary_iter)
    secondary_iter = iterate_eternally(secondary_iter,is_shuffle=False)
    return (
        secondary_batch + primary_batch
        for (primary_batch, secondary_batch)
        in zip(grouper(primary_iter, 10),
               grouper(secondary_iter, 3))
    )

