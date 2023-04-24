import csv
import itertools
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, RegressorMixin
import pandas as pd
import numpy as np
from prediction.utils.CONST_VAR import VAR_LIST, DICT_Y
from prediction.utils.metrics import cal_metrics
from prediction.utils.noise import cos_sim_noise
from prediction.utils.smooth_data import savgol_filter_A
from prediction.utils.split_dataset import split_by_bin, split_by_order
from prediction.utils.visualization import draw_pred_curve


class Mydataset(Dataset):

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


class LongShortTermMemory(nn.Module):
    # Initialization
    def __init__(self, dim_X, dim_y, lstm=(1024,)):
        super(LongShortTermMemory, self).__init__()

        # Parameter assignment
        self.dim_X = dim_X
        self.dim_y = dim_y
        self.net_lstm = [dim_X, ] + list(lstm)
        self.net_fc = [lstm[-1], 1]

        # Model creation
        self.lstm = nn.ModuleList()
        self.fc = nn.ModuleList()
        for i in range(dim_y):
            self.lstm.append(nn.ModuleList())
            for j in range(len(lstm)):
                self.lstm[-1].append(nn.LSTM(self.net_lstm[j], self.net_lstm[j + 1]))
            self.fc.append(nn.Linear(self.net_fc[0], self.net_fc[1]))

    # Forward propagation
    def forward(self, X):
        res_list = []
        for i in range(self.dim_y):
            feat = X
            for j in self.lstm[i]:
                feat = j(feat)[0]
            feat = self.fc[i](feat)
            res_list.append(feat.squeeze())
        res = torch.stack(res_list, dim=-1)

        return res


class LSTMModel(BaseEstimator, RegressorMixin):

    def __init__(self, dim_X, dim_y, lstm=(1024,), seq_len=30, n_epoch=200, batch_size=64, lr=0.001, weight_decay=0.1,
                 step_size=50, gamma=0.5, gpu=torch.device('cuda:2'), seed=1024):
        super(LSTMModel, self).__init__()

        # Set seed
        torch.manual_seed(seed)

        # Parameter assignment
        self.dim_X = dim_X
        self.dim_y = dim_y
        self.lstm = lstm
        self.seq_len = seq_len
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.step_size = step_size
        self.gamma = gamma
        self.gpu = gpu
        self.seed = seed

        # Initialize scaler
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

        # Model creation
        self.loss_hist = []
        self.model = LongShortTermMemory(dim_X, dim_y, lstm).to(gpu)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size, gamma)
        self.criterion = nn.MSELoss(reduction='sum')

    def fit(self, X, y):
        def count_param(model):
            param_count = 0
            for param in model.parameters():
                param_count += param.view(-1).size()[0]
            return param_count

        X = self.scaler_X.fit_transform(X)
        y = self.scaler_y.fit_transform(y)
        X_3d = []
        y_3d = []
        for i in range(X.shape[0] - self.seq_len + 1):
            X_3d.append(X[i: i + self.seq_len, :])
            y_3d.append(y[i: i + self.seq_len, :])

        X_3d = np.stack(X_3d, 1)
        y_3d = np.stack(y_3d, 1)
        dataset = Mydataset(torch.tensor(X_3d, dtype=torch.float32, device=self.gpu),
                            torch.tensor(y_3d, dtype=torch.float32, device=self.gpu), '3D')

        self.model.train()
        for i in range(self.n_epoch):
            self.loss_hist.append(0)
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            for batch_X, batch_y in data_loader:
                batch_X = batch_X.permute(1, 0, 2)
                batch_y = batch_y.permute(1, 0, 2)
                self.optimizer.zero_grad()
                output = self.model(batch_X)
                loss = self.criterion(output, batch_y)
                self.loss_hist[-1] += loss.item()
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()
            print('Epoch:{}, Loss:{}'.format(i + 1, self.loss_hist[-1]))
        print('Optimization finished')

        return self

    def predict(self, X):
        X = torch.tensor(self.scaler_X.transform(X), dtype=torch.float32, device=self.gpu).unsqueeze(1)
        self.model.eval()
        with torch.no_grad():
            y = self.scaler_y.inverse_transform(self.model(X).cpu().numpy())
        return y, self.scaler_y


# ******************************************************************
data_path = r'../A2_data/202201/512/1月512有标签.csv'
shuffle = False
add_noise = False
smooth_data = False
window_size = 31
poly_order = 5
bin_width = 1
TRAIN_RATIO = 0.8

net_strcut = [25, 23, 18, 14]
y_index = DICT_Y['K+']
save_path = None
show_plot = True
train = True
device = 'cpu'
# ******************************************************************
df = pd.read_csv(data_path, encoding='utf-8-sig')
data = np.array(df[VAR_LIST[1:-1]])
var_num = DICT_Y['K+']

if smooth_data:
    data[:, :var_num] = savgol_filter_A(data[:, :var_num], window_size=window_size, poly_order=poly_order)

if True:
    # 顺序划分
    dataset, data_size = split_by_order(data[:, :var_num], data[:, y_index], shuffle=shuffle, seed=1024)
    # 按区间划分数据集
    # dataset, data_size = split_by_bin(data[:, :var_num], data[:, y_index], bin_width=bin_width, inside_shuffle=shuffle,
    #                                   seed=1024)
    train_data, val_data, test_data = dataset
    train_num, val_num, test_num = data_size

if train:
    train_X = train_data[:, 0:var_num]
    train_y = train_data[:, -1].reshape(-1, 1)

    val_X = val_data[:, 0:var_num]
    val_y = val_data[:, -1].reshape(-1, 1)
else:
    train_X = np.append(train_data[:, 0:var_num], val_data[:, :var_num], axis=0)
    train_y = np.append(train_data[:, -1].reshape(-1, 1), val_data[:, -1].reshape(-1, 1), axis=0)

    val_X = test_data[:, 0:var_num]
    val_y = test_data[:, -1].reshape(-1, 1)

if add_noise:
    train_X, train_y = cos_sim_noise(train_X, train_y, bias_factor=2)

print('训练集：--------')
print(train_X.shape)
print(train_y.shape)
if train:
    print('验证集：--------')
else:
    print('测试集：--------')
print(val_X.shape)
print(val_y.shape)

mdl = LSTMModel(train_X.shape[1], train_y.shape[1], lstm=(256,), seq_len=40, n_epoch=20, batch_size=128,
                lr=0.001, weight_decay=0.1,
                step_size=50, gamma=0.5, gpu=torch.device('cpu'), seed=1024).fit(train_X, train_y)

# 训练集
pred_train_y, _ = mdl.predict(train_X)
train_res = cal_metrics(y_true=train_y, y_pred=pred_train_y)

# 测试集
pred_val_y, _ = mdl.predict(val_X)
val_res = cal_metrics(y_true=val_y, y_pred=pred_val_y)

if show_plot:
    draw_pred_curve(train_y, pred_train_y, title='train_set', fig_size=(14, 10), save_path=save_path)
    draw_pred_curve(val_y, pred_val_y, title='test_set', fig_size=(14, 10), save_path=save_path)
