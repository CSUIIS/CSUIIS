import numpy as np
import random


def split_by_order(data_X, data_y, train_ratio=0.8, shuffle=False, seed=None):
    assert data_X.shape[0] == data_y.shape[0]
    data_xy = np.append(data_X, data_y.reshape(-1, 1), axis=1)
    if shuffle:
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(data_xy)

    train_num = int(data_xy.shape[0] * train_ratio)
    val_num = (data_xy.shape[0] - train_num) // 2

    train_data = data_xy[:train_num, :]
    val_data = data_xy[train_num:train_num + val_num, :]
    test_data = data_xy[train_num + val_num:, :]

    dataset = [train_data, val_data, test_data]
    data_size = [train_data.shape[0], val_data.shape[0], test_data.shape[0]]
    return dataset, data_size


def split_by_bin(data_X, data_y, bin_width=1, train_ratio=0.8, inside_shuffle=False, seed=1024, plot=False):
    assert data_X.shape[0] == data_y.shape[0]

    min_y = np.min(data_y)
    max_y = np.max(data_y)

    y_start = int(min_y)
    y_end = y_start + (int((max_y - y_start) / bin_width) + 1) * bin_width

    bin_num = int((y_end - y_start) / bin_width)

    data_xy = np.append(data_X, data_y.reshape(-1, 1), axis=1)

    data_each_bin = []
    for i in range(bin_num):
        data_each_bin.append([])

    # 遍历所有数据，根据标签值放入对应的区间   左闭右开
    for i in range(data_xy.shape[0]):
        bin_index = int((data_xy[i][-1] - y_start) / bin_width)
        row_data = data_xy[i, :]
        data_each_bin[bin_index].append(list(row_data))

    if seed is not None:
        random.seed(seed)
    random.shuffle(data_each_bin)
    if inside_shuffle:
        for i in range(len(data_each_bin)):
            random.shuffle(data_each_bin[i])

    if plot:
        print('训练集比例为{}，窗宽为{}'.format(train_ratio, bin_width))
        import matplotlib.pyplot as plt
        import math
        total_data = None
        pos_split = []
        pos_bin = []
        for i in range(len(data_each_bin)):
            bin_data = np.array(data_each_bin[i])
            train_num = int(bin_data.shape[0] * train_ratio)
            val_num = (bin_data.shape[0] - train_num) // 2

            if i == 0:
                pos_split.append(train_num + val_num)
                total_data = bin_data
                pos_bin.append(total_data.shape[0])
            else:
                pos_split.append(total_data.shape[0] + train_num + val_num)
                total_data = np.append(total_data, bin_data, axis=0)
                pos_bin.append(total_data.shape[0])

        total_y = total_data[:, -1]
        plt.figure(figsize=(20,10))
        plt.plot(total_y)
        for i in range(math.floor(np.min(total_y)), math.ceil(np.max(total_y))):
            plt.axhline(y=i, ls="-", c="black")
        for i in range(len(pos_split)):
            plt.axvline(x=pos_split[i],ls=":",c="red")
        for i in range(len(pos_bin)):
            plt.axvline(x=pos_bin[i], ls="-", c="black")
        plt.show()
        exit(0)

    train_data = None
    val_data = None
    test_data = None
    for i in range(len(data_each_bin)):
        if len(data_each_bin[i]) == 0:      # 有些区间可能没有数据
            continue
        bin_data = np.array(data_each_bin[i])
        train_num = int(bin_data.shape[0] * train_ratio)
        val_num = (bin_data.shape[0] - train_num) // 2

        if i == 0:
            train_data = bin_data[0:train_num, :]
            val_data = bin_data[train_num:train_num + val_num, :]
            test_data = bin_data[train_num + val_num:, :]

        else:
            train_data = np.append(train_data, bin_data[0:train_num, :], axis=0)
            val_data = np.append(val_data, bin_data[train_num:train_num + val_num, :], axis=0)
            test_data = np.append(test_data, bin_data[train_num + val_num:, :], axis=0)

    dataset = [train_data, val_data, test_data]
    data_size = [train_data.shape[0], val_data.shape[0], test_data.shape[0]]
    return dataset, data_size


def split_by_label(data_X, data_y, train_size, index_column=0, shuffle=False, seed=None):
    assert train_size > 20
    assert data_X.shape[0] == data_y.shape[0]
    assert data_y.shape[1] == 5, print('应传入5个离子浓度')

    data_xy = np.append(data_X, data_y[:, index_column].reshape(-1, 1), axis=1)

    y_list = []
    for i in range(data_y.shape[0]):
        if list(data_y[i, :]) not in y_list:
            y_list.append(list(data_y[i, :]))

    data_by_y = []
    for i in range(len(y_list)):
        data_by_y.append([])

    for i in range(data_X.shape[0]):
        index = y_list.index(list(data_y[i, :]))
        data_by_y[index].append(list(data_xy[i, :]))

    if shuffle:
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(data_by_y)

    train_data = None
    val_data = None
    test_data = None
    for i in range(train_size - 10):
        if train_data is None:
            train_data = np.array(data_by_y[i])
        else:
            train_data = np.append(train_data, np.array(data_by_y[i]), axis=0)
    for i in range(train_size - 10, train_size):
        if val_data is None:
            val_data = np.array(data_by_y[i])
        else:
            val_data = np.append(val_data, np.array(data_by_y[i]), axis=0)
    for i in range(train_size, len(data_by_y)):
        if test_data is None:
            test_data = np.array(data_by_y[i])
        else:
            test_data = np.append(test_data, np.array(data_by_y[i]), axis=0)

    dataset = [train_data, val_data, test_data]
    data_size = [train_data.shape[0], val_data.shape[0], test_data.shape[0]]
    return dataset, data_size, data_by_y


def split_by_bin_v2(data_X, data_y, bin_width=1, train_ratio=0.8, inside_shuffle=False, seed=1024):
    assert data_X.shape[0] == data_y.shape[0]
    total_y = data_y.copy()
    data_y = data_y[:,0]

    min_y = np.min(data_y)
    max_y = np.max(data_y)

    y_start = int(min_y)
    y_end = y_start + (int((max_y - y_start) / bin_width) + 1) * bin_width

    bin_num = int((y_end - y_start) / bin_width)

    data_xy = np.append(data_X, total_y, axis=1)

    data_each_bin = []
    for i in range(bin_num):
        data_each_bin.append([])

    # 遍历所有数据，根据标签值放入对应的区间   左闭右开
    for i in range(data_xy.shape[0]):
        bin_index = int((data_y[i] - y_start) / bin_width)
        row_data = data_xy[i, :]
        data_each_bin[bin_index].append(list(row_data))

    if seed is not None:
        random.seed(seed)
    random.shuffle(data_each_bin)
    if inside_shuffle:
        for i in range(len(data_each_bin)):
            random.shuffle(data_each_bin[i])

    train_data = None
    val_data = None
    test_data = None
    for i in range(len(data_each_bin)):
        bin_data = np.array(data_each_bin[i])
        train_num = int(bin_data.shape[0] * train_ratio)
        val_num = (bin_data.shape[0] - train_num) // 2

        if i == 0:
            train_data = bin_data[0:train_num, :]
            val_data = bin_data[train_num:train_num + val_num, :]
            test_data = bin_data[train_num + val_num:, :]

        else:
            train_data = np.append(train_data, bin_data[0:train_num, :], axis=0)
            val_data = np.append(val_data, bin_data[train_num:train_num + val_num, :], axis=0)
            test_data = np.append(test_data, bin_data[train_num + val_num:, :], axis=0)

    dataset = [train_data, val_data, test_data]
    data_size = [train_data.shape[0], val_data.shape[0], test_data.shape[0]]
    return dataset, data_size