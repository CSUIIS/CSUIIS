import torch
import torch.functional as F
import numpy as np
import pandas as pd
import scipy.signal
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from prediction.utils.CONST_VAR_SERVER import VAR_LIST, DICT_Y, DataPath
from prediction.utils.metrics import cal_metrics
from prediction.NN.dataset import MyDataset_NN
import os
from prediction.NN.model import TestNet
from tqdm import tqdm
import sys
from prediction.utils.noise import *
from prediction.utils.smooth_data import savgol_filter_A
from prediction.utils.split_dataset import *
from prediction.utils.visualization import draw_pred_curve


def weight_mse(pred_y, real_y):
    weight = []
    for i in range(real_y.size()[0]):
        down_bound = int(real_y[i][0])
        up_bound = int(real_y[i][0]) + 1
        temp = train_y[train_y >= down_bound]
        temp = temp[temp <= up_bound]
        ratio = temp.shape[0] / train_y.shape[0]
        weight.append(1 / ratio)
    weight = np.array(weight).reshape((-1, 1))
    weight = MinMaxScaler().fit_transform(weight)
    weight = weight + 0.25

    weight = torch.tensor(weight)
    error = (pred_y - real_y) ** 2
    error = error * weight
    loss = torch.sum(error) / pred_y.size()[0]
    return loss


def weighted_mse_loss(inputs, targets, weights=None):
    loss = (inputs - targets) ** 2
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_l1_loss(inputs, targets, weights=None):
    loss = F.l1_loss(inputs, targets, reduction='none')
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_focal_mse_loss(inputs, targets, weights=None, activate='sigmoid', beta=.2, gamma=1):
    loss = (inputs - targets) ** 2
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_focal_l1_loss(inputs, targets, weights=None, activate='sigmoid', beta=.2, gamma=1):
    loss = F.l1_loss(inputs, targets, reduction='none')
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_huber_loss(inputs, targets, weights=None, beta=1.):
    l1_loss = torch.abs(inputs - targets)
    cond = l1_loss < beta
    loss = torch.where(cond, 0.5 * l1_loss ** 2 / beta, l1_loss - 0.5 * beta)
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


# ******************************************************************
data_path = DataPath.m1_512
exp_num = '7-6'
shuffle = False
add_noise = False
smooth_data = False
window_size = 31
poly_order = 5
y_index = DICT_Y['K+']
save_plot = False
show_plot = False

bin_width = 1
batch_size = 128
lr = 0.001
epochs = 300
# momentum = 0.9
device = 'cpu'
layers = [25, 23, 18, 14]
dropout = None
# ******************************************************************
df = pd.read_csv(data_path, encoding='utf-8-sig')
data = np.array(df[VAR_LIST[1:-1]])
var_num = DICT_Y['K+']

if smooth_data:
    # 整体平滑
    data[:, :var_num] = savgol_filter_A(data[:, :var_num], window_size=window_size, poly_order=poly_order)

if True:
    # 顺序划分
    dataset, data_size = split_by_order(data[:,:var_num],data[:,y_index], shuffle=shuffle)
    # 按区间划分数据集
    # dataset, data_size = split_by_bin(data[:, :var_num], data[:, y_index], bin_width=bin_width,inside_shuffle=shuffle)
    train_data, val_data,test_data = dataset
    train_num, val_num,test_num = data_size

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

train_X = train_data[:, :var_num]
train_X = scaler_X.fit_transform(train_X)
train_y = train_data[:, -1].reshape(-1, 1)
train_y = scaler_y.fit_transform(train_y)

if add_noise:
    # 按照与prototy的距离加噪
    train_X, train_y = cos_sim_noise(train_X, train_y, bias_factor=2)

val_X = val_data[:, :var_num]
val_X = scaler_X.transform(val_X)
val_y = val_data[:, -1].reshape(-1, 1)
val_y = scaler_y.transform(val_y)

train_set = MyDataset_NN(torch.tensor(train_X, dtype=torch.float32), torch.tensor(train_y, dtype=torch.float32))
# num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
num_workers = 0
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

print("Using {} images for train".format(train_num))
print("Using {} images for validation".format(val_num))
print("Using {} images for test".format(test_num))
model = TestNet(layers, dropout=dropout)
model.to(device)
# loss_function = weight_mse
loss_function = nn.MSELoss()

# optimizer = optim.Adam(model.parameters(), lr=lr)
optimizer = optim.SGD(model.parameters(),lr=lr,momentum=0.9,weight_decay=5e-4,nesterov=True)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)

save_dir = './model'
model_num = len(os.listdir('./model')) - 1
train_steps = len(train_loader)

model.train()
for epoch in range(epochs):
    epoch_loss = 0.0
    train_bar = tqdm(train_loader, file=sys.stdout)
    # scheduler.step()
    for step, data in enumerate(train_bar):
        batch_X, batch_y = data
        # exit(0)
        outputs = model(batch_X.to(device))
        loss = loss_function(outputs, batch_y.to(device))
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += loss.item()
        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                 epochs,
                                                                 loss)
    print("train epoch[{}] running_loss:{:.3f}".format(epoch + 1, epoch_loss))
torch.save(model.state_dict(), os.path.join(save_dir, 'testnet_{}_final.pth'.format(model_num + 1)))
print('Finished Training')

model.eval()
train_pred_y = model(torch.tensor(train_X, dtype=torch.float32, device=device)).detach().numpy()
test_pred_y = model(torch.tensor(val_X, dtype=torch.float32, device=device)).detach().numpy()

# 反归一化
# train_y = scaler_y.inverse_transform(train_y)
# val_y = scaler_y.inverse_transform(val_y)
# train_pred_y = scaler_y.inverse_transform(train_pred_y)
# test_pred_y = scaler_y.inverse_transform(test_pred_y)

cal_metrics(y_true=train_y, y_pred=train_pred_y)
cal_metrics(y_true=val_y, y_pred=test_pred_y)

# 训练集画图
if show_plot:
    draw_pred_curve(train_y,train_pred_y,title='train_set')
    draw_pred_curve(val_y,test_pred_y,title='test_set')


