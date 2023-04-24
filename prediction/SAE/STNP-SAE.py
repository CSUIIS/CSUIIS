import csv

import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import MinMaxScaler
from prediction.utils.CONST_VAR import VAR_LIST, DICT_Y
from prediction.utils.metrics import cal_metrics
from prediction.utils.noise import *
from prediction.utils.smooth_data import *
from prediction.utils.split_dataset import *
import itertools
import time as realtime
import heapq
from datetime import datetime
import argparse
from prediction.utils.visualization import draw_pred_curve
import copy

"""
质量监督堆叠自编码器的代码
"""


def save_log(config, path):
    with open(path, 'w', encoding='utf-8-sig') as f:
        for arg in vars(config):
            log_info = format(arg, '<20') + format(str(getattr(config, arg)), '<') + '\n'
            f.write(log_info)


def SNP_TNPLoss(x_recon, idxs, total_X, knn_index, M):
    loss = 0.
    for i in range(x_recon.size()[0]):
        # nn_idxs = knn_index[idxs[i], :]
        nn_idxs = knn_index[idxs[i], :].astype('int64')
        weight = torch.tensor(M[idxs[i], :])

        nn_array = torch.tensor(total_X[nn_idxs, :])
        error = torch.linalg.norm(nn_array - x_recon[i, :], axis=1)
        loss += torch.sum(error ** 2 * weight)
    loss /= x_recon.size()[0] * knn_index.shape[1]
    return loss


def SNP_TNPLoss_v2(x_recon, knn_output, w):
    bs = x_recon.size()[0]
    k = knn_output.size()[1]
    x_recon = x_recon.reshape(bs, 1, -1)
    res = torch.square(knn_output - x_recon)
    res = torch.sum(res, dim=2).reshape(-1)
    loss = torch.sum(res * w)
    loss /= bs * k
    return loss


def EuclideanDistances(A, B, sqrt=False):
    BT = B.transpose()
    # vecProd = A * BT
    vecProd = np.dot(A, BT)
    # print(vecProd)
    SqA = A ** 2
    # print(SqA)
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))
    # print(sumSqAEx)

    SqB = B ** 2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))
    SqED = sumSqBEx + sumSqAEx - 2 * vecProd
    SqED[SqED < 0] = 0.0
    if sqrt:
        ED = np.sqrt(SqED)
    else:
        ED = SqED
    ED = np.array(ED)
    return ED

def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count

# 数据集定义方式
class MyDataset(Dataset):

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


class MyDataset_unsup(Dataset):

    # Initialization
    def __init__(self, data, mode='2D'):
        self.data, self.mode = data, mode

    # Get item
    def __getitem__(self, index):
        if self.mode == '2D':
            return self.data[index, :], index
        elif self.mode == '3D':
            return self.data[:, index, :]

    # Get length
    def __len__(self):
        if self.mode == '2D':
            return self.data.shape[0]
        elif self.mode == '3D':
            return self.data.shape[1]


# 自编码器的定义
class AutoEncoder(nn.Module):
    def __init__(self, dim_X, dim_H):
        super(AutoEncoder, self).__init__()
        self.dim_X = dim_X
        self.dim_H = dim_H
        self.act = torch.sigmoid

        self.encoder = nn.Linear(dim_X, dim_H, bias=True)
        self.decoder = nn.Linear(dim_H, dim_X, bias=True)

    def forward(self, X, rep=False):
        H = self.act(self.encoder(X))
        if rep is False:
            return self.act(self.decoder(H))
        else:
            return H


# 堆叠自编码器定义生成
class StackedAutoEncoder(nn.Module):
    def __init__(self, size, device=torch.device('cuda:0')):
        super(StackedAutoEncoder, self).__init__()
        self.AElength = len(size)
        self.SAE = []
        self.device = device

        for i in range(1, self.AElength):
            self.SAE.append(AutoEncoder(size[i - 1], size[i]).to(device))

        self.proj = nn.Linear(size[self.AElength - 1], 1)

    def forward(self, X, NoL, PreTrain=False):
        """
        :param X: 进口参数
        :param NoL: 第几层
        :param PreTrain: 是不是无监督预训练
        :return:
        """
        out = X
        if PreTrain is True:
            # SAE的预训练
            if NoL == 0:
                return out, self.SAE[NoL](out)

            else:
                for i in range(NoL):
                    # 第N层之前的参数给冻住
                    for param in self.SAE[i].parameters():
                        param.requires_grad = False

                    out = self.SAE[i](out, rep=True)
                # 训练第N层
                inputs = out
                out = self.SAE[NoL](out)
                return inputs, out
        else:
            for i in range(self.AElength - 1):
                # 做微调
                for param in self.SAE[i].parameters():
                    param.requires_grad = True

                out = self.SAE[i](out, rep=True)
            # out = torch.sigmoid(self.proj(out))
            out = torch.tanh(self.proj(out))
            # out = self.proj(out)
            return out


# 单层自编码器训练函数
def trainAE(w1, w2, model, trainloader, epochs, trainlayer, lr,
            total_X, s_knn_index, M_s,
            t_knn_index, M_t):
    s_knn_index = torch.from_numpy(s_knn_index).to(torch.long)
    t_knn_index = torch.from_numpy(t_knn_index).to(torch.long)
    M_s = torch.from_numpy(M_s).to(torch.float32)
    M_t = torch.from_numpy(M_t).to(torch.float32)
    total_X = torch.from_numpy(total_X).to(torch.float32)

    optimizer = torch.optim.Adam(model.SAE[trainlayer].parameters(), lr=lr)
    # loss_func = nn.MSELoss()
    loss_func = SNP_TNPLoss

    for ep in range(epochs):
        sum_loss = 0
        for X, idxs in trainloader:
            # Hidden 输入 Hidden_recon 输出
            Hidden, Hidden_recon = model(X, trainlayer, PreTrain=True)
            # loss = loss_func(Hidden, Hidden_recon)

            # -------------------------
            bs = X.size()[0]
            spa_knn_idx = s_knn_index[idxs, :].reshape(-1)
            tmp_knn_idx = t_knn_index[idxs, :].reshape(-1)

            spa_knn_output = total_X[spa_knn_idx, :].reshape(bs, s_knn_index.shape[1], -1)
            tem_knn_output = total_X[tmp_knn_idx, :].reshape(bs, t_knn_index.shape[1], -1)

            spa_w = M_s[idxs, :].reshape(-1)
            tem_w = M_t[idxs, :].reshape(-1)

            snp_loss = SNP_TNPLoss_v2(Hidden_recon, spa_knn_output, spa_w)
            tnp_loss = SNP_TNPLoss_v2(Hidden_recon, tem_knn_output, tem_w)

            loss = w1 * snp_loss + w2 * tnp_loss
            # -------------------------
            # total_X = total_X.numpy()
            # s_knn_index = s_knn_index.numpy()
            # t_knn_index = t_knn_index.numpy()
            # M_s = M_s.numpy()
            # M_t = M_t.numpy()
            #
            # loss1 = loss_func(Hidden_recon, idxs, total_X, s_knn_index, M_s)
            # loss2 = loss_func(Hidden_recon, idxs, total_X, t_knn_index, M_t)
            # loss = w1 * loss1 + w2 * loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += loss.detach().item()
        # print('无监督预训练第{}层的第{}个epoch, '.format(trainlayer + 1, ep + 1), ',其Loss的大小是:{}'.format(loss.data.cpu().numpy()))

    return model


# SAE训练的代码模型
class SAEModel(BaseEstimator, RegressorMixin):
    def __init__(self, config):
        super(SAEModel, self).__init__()
        torch.manual_seed(config.seed)

        self.config = config
        # 参数分配
        self.AEList = config.AEList
        self.num_AE = len(config.AEList) - 1
        self.unsp_epoch = config.unsp_epoch
        self.sup_epoch = config.sup_epoch
        self.unsp_batch_size = config.unsp_batch_size
        self.sp_batch_size = config.sp_batch_size
        self.unsp_lr = config.unsp_lr
        self.sp_lr = config.sp_lr
        self.device = config.device
        self.seed = config.seed

        self.scaler_X = MinMaxScaler()
        self.total_X = None

        self.K_s = config.K_s
        self.spatial_knn_index = None  # 存放K近邻的索引，不按顺序
        self.spatial_M = None  # 存放K近邻的损失权重，顺序同self.knn_index
        self.delta_s = config.delta_s

        self.K_t = config.K_t
        self.temporal_knn_index = None
        self.temporal_M = None
        self.delta_t = config.delta_t

        self.snp_loss_weight = config.snp_loss_weight
        self.tnp_loss_weight = config.tnp_loss_weight

        self.res_dir_path = config.res_dir_path

        # SAE模型的创建
        self.StackedAutoEncoderModel = StackedAutoEncoder(size=config.AEList, device=config.device).to(config.device)

        # 有多少AE就要单独定义多少次SAE
        self.optimizer = optim.Adam(
            [
                {'params': self.StackedAutoEncoderModel.parameters(), 'lr': self.unsp_lr},
                {'params': self.StackedAutoEncoderModel.SAE[0].parameters(), 'lr': self.sp_lr},
                {'params': self.StackedAutoEncoderModel.SAE[1].parameters(), 'lr': self.sp_lr},
                {'params': self.StackedAutoEncoderModel.SAE[2].parameters(), 'lr': self.sp_lr}
            ])

        self.loss_func = nn.MSELoss()
        # self.loss_func = nn.L1Loss()
        # self.loss_func = weight_mse
        self.SAEModel_150 = None
        self.SAEModel_200 = None
        self.SAEModel_250 = None

    # 数据拟合
    def fit(self, labeled_X, labeled_y, unlabeled_X):
        labeled_X = labeled_X[::, :]
        labeled_y = labeled_y[::, :]
        unlabeled_X = unlabeled_X[::self.config.unlab_ratio, :]
        print('实际使用无标签：', unlabeled_X.shape)

        total_X = np.append(labeled_X, unlabeled_X, axis=0)
        # total_X = labeled_X
        print('实际总数据：', total_X.shape)
        print('-------------------------')
        time_series = total_X[:, 0]

        total_X = total_X[:, 1:]
        total_X = self.scaler_X.fit_transform(total_X)
        self.total_X = total_X

        labeled_X = labeled_X[:, 1:]
        labeled_X = self.scaler_X.transform(labeled_X)

        total_X = torch.tensor(total_X, dtype=torch.float32, device=self.device)
        unsup_dataset = MyDataset_unsup(total_X, '2D')
        sup_dataset = MyDataset(torch.tensor(labeled_X, dtype=torch.float32, device=self.device),
                                torch.tensor(labeled_y, dtype=torch.float32, device=self.device), '2D')

        un_trainloader = DataLoader(unsup_dataset, batch_size=self.unsp_batch_size, shuffle=True)
        trainloader = DataLoader(sup_dataset, batch_size=self.sp_batch_size, shuffle=True)
        print(next(self.StackedAutoEncoderModel.parameters()).is_cuda)
        self.StackedAutoEncoderModel.train()

        for i in range(self.num_AE):
            print('自编码器训练第{}层:'.format(i + 1))
            self.StackedAutoEncoderModel.SAE[i].eval()
            if i == 0:
                tem_dir_path = os.path.join('../MT_TSNLNet/近邻矩阵/512/temporal_mat',
                                            '{}_{}_{}'.format(self.total_X.shape[0], self.K_t, self.delta_t))

                if os.path.exists(tem_dir_path):
                    print('载入储存的[时间]近邻权重矩阵...')
                    df1 = pd.read_csv(os.path.join(tem_dir_path, 'temporal_knn_index.csv'))
                    self.temporal_knn_index = np.array(df1)[:, 1:]
                    df2 = pd.read_csv(os.path.join(tem_dir_path, 'temporal_M.csv'))
                    self.temporal_M = np.array(df2)[:, 1:]
                else:
                    print('未存储当前[时间]近邻权重！开始计算...')

                    for j in range(time_series.shape[0]):
                        time_series[j] = datetime.timestamp(time_series[j])
                        time_series[j] -= 1641542340.
                    time_series = time_series / 86400
                    time_series = time_series.reshape(-1, 1)
                    scaler_t = MinMaxScaler()
                    time_series = scaler_t.fit_transform(time_series)

                    # 计算时间邻接图和权重矩阵
                    self.temporal_M = np.zeros((total_X.shape[0], self.K_t))
                    self.temporal_knn_index = np.zeros((total_X.shape[0], self.K_t))
                    for j in range(self.total_X.shape[0]):
                        dist = time_series - time_series[j]
                        dist = np.abs(dist).reshape(-1)

                        # 最小K+1个数的索引，不一定按顺序
                        flat_indices = np.argpartition(dist, self.K_t)[:self.K_t + 1]
                        row_indices = np.unravel_index(flat_indices, dist.shape)[0]
                        # 返回距离最近K的样本点的索引，其中第K个索引是距离第K近，前K-1个索引不一定按顺序排列
                        row_indices = row_indices[row_indices != j]
                        self.temporal_knn_index[j, :] = row_indices
                        select_dist = dist[row_indices]
                        val = ((select_dist / np.sum(select_dist)) ** 2).astype('float')
                        self.temporal_M[j, :] = np.exp(-val / self.delta_t)

                    os.mkdir(tem_dir_path)
                    df1 = pd.DataFrame(self.temporal_knn_index)
                    df1.to_csv(os.path.join(tem_dir_path, 'temporal_knn_index.csv'))
                    df2 = pd.DataFrame(self.temporal_M)
                    df2.to_csv(os.path.join(tem_dir_path, 'temporal_M.csv'))
                    print('存入[{}]...'.format(tem_dir_path))

                spa_dir_path = os.path.join('../MT_TSNLNet/近邻矩阵/512/spatial_mat',
                                            '{}_{}_{}_{}'.format(total_X.shape[0], self.K_s, self.delta_s,
                                                                 self.config.smooth_data))
                if os.path.exists(spa_dir_path):
                    print('载入存储的[空间]近邻权重矩阵...')
                    df1 = pd.read_csv(os.path.join(spa_dir_path, 'spatial_knn_index.csv'))
                    self.spatial_knn_index = np.array(df1)[:, 1:]
                    df2 = pd.read_csv(os.path.join(spa_dir_path, 'spatial_M.csv'))
                    self.spatial_M = np.array(df2)[:, 1:]
                else:
                    print('未存储当前[空间]近邻权重！开始计算...')

                    ## 计算空间邻接图和权重矩阵
                    self.spatial_knn_index = np.zeros((total_X.shape[0], self.K_s))
                    self.spatial_M = np.zeros((total_X.shape[0], self.K_s))
                    dis_mat = None
                    batch_idx = 0
                    bs = 1000
                    for j in range(self.total_X.shape[0]):
                        if j % bs == 0:
                            dis_mat = EuclideanDistances(self.total_X[batch_idx * bs:(batch_idx + 1) * bs, :],
                                                         self.total_X)
                            batch_idx += 1

                        dist = dis_mat[j - (batch_idx - 1) * bs, :].copy().reshape(-1)
                        # 最小K+1个数的索引，不一定按顺序
                        flat_indices = np.argpartition(dist, self.K_s)[:self.K_s + 1]
                        row_indices = np.unravel_index(flat_indices, dist.shape)[0]

                        # 返回距离最近K的样本点的索引，其中第K个索引是距离第K近，前K-1个索引不一定按顺序排列
                        row_indices = row_indices[row_indices != j]
                        self.spatial_knn_index[j, :] = row_indices
                        self.spatial_M[j, :] = np.exp(-dist[row_indices] / self.delta_s)

                    os.mkdir(spa_dir_path)
                    df1 = pd.DataFrame(self.spatial_knn_index)
                    df1.to_csv(os.path.join(spa_dir_path, 'spatial_knn_index.csv'))
                    df2 = pd.DataFrame(self.spatial_M)
                    df2.to_csv(os.path.join(spa_dir_path, 'spatial_M.csv'))
                    print('存入[{}]...'.format(spa_dir_path))

            self.StackedAutoEncoderModel.SAE[i].train()
            self.StackedAutoEncoderModel = trainAE(self.snp_loss_weight, self.tnp_loss_weight,
                                                   model=self.StackedAutoEncoderModel, trainloader=un_trainloader,
                                                   epochs=self.unsp_epoch, trainlayer=i, lr=self.unsp_lr,
                                                   total_X=self.total_X, s_knn_index=self.spatial_knn_index,
                                                   M_s=self.spatial_M,
                                                   t_knn_index=self.temporal_knn_index, M_t=self.temporal_M)
            self.StackedAutoEncoderModel.SAE[i].eval()
            self.total_X = self.StackedAutoEncoderModel.SAE[i](
                torch.tensor(self.total_X, dtype=torch.float32, device=self.device)
                , rep=True).detach().numpy()
            print('自编码器第{}层训练完成!'.format(i + 1))

        Loss = []
        for i in range(self.sup_epoch):
            sum_loss = 0
            for batch_X, batch_y in trainloader:
                pre = self.StackedAutoEncoderModel(batch_X, i, PreTrain=False)
                loss = self.loss_func(pre, batch_y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                sum_loss += loss.detach().item()
            # print('有监督微调第{}轮的Loss是{}'.format(i + 1, sum_loss))
            Loss.append(sum_loss)
            if (i + 1) % self.config.save_freq == 0:
                if not os.path.exists(self.res_dir_path):
                    os.mkdir(self.res_dir_path)
                    save_log(self.config, os.path.join(self.res_dir_path, 'sup_para.txt'))

                torch.save(self.StackedAutoEncoderModel.state_dict(),
                           os.path.join(self.res_dir_path, 'STNP_SAE_epoch{}.pkl'.format(i)))
            if (i + 1) == 150:
                self.SAEModel_150 = copy.deepcopy(self.StackedAutoEncoderModel)
            elif (i + 1) == 200:
                self.SAEModel_200 = copy.deepcopy(self.StackedAutoEncoderModel)
            elif (i + 1) == 250:
                self.SAEModel_250 = copy.deepcopy(self.StackedAutoEncoderModel)
        if self.sup_epoch >= self.config.save_freq:
            torch.save(self.StackedAutoEncoderModel.state_dict(),
                       os.path.join(self.res_dir_path, 'STNP_SAE_epoch{}.pkl'.format(config.sup_epoch - 1)))
        # 绘制损失函数曲线
        plt.figure()
        plt.plot(range(len(Loss)), Loss, color='b')
        plt.savefig(os.path.join(self.res_dir_path, 'loss.png'))
        plt.show()

        return self

    # 预测数据
    def predict(self, X):
        # X = self.scaler_X.transform(X)[0].reshape(1,-1)
        # X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        #
        # self.SAEModel_150.eval()
        # with torch.no_grad():
        #     import time
        #     a = time.clock()
        #     for i in range(950):
        #         y_150 = self.SAEModel_150(X, 0, PreTrain=False).cpu().numpy()
        #     b = time.clock()
        #     print(b - a)
        #     exit(0)
        # self.SAEModel_200.eval()
        # with torch.no_grad():
        #     y_200 = self.SAEModel_200(X, 0, PreTrain=False).cpu().numpy()
        # self.SAEModel_250.eval()
        # with torch.no_grad():
        #     y_250 = self.SAEModel_250(X, 0, PreTrain=False).cpu().numpy()
        # self.StackedAutoEncoderModel.eval()
        # with torch.no_grad():
        #     y = self.StackedAutoEncoderModel(X, 0, PreTrain=False).cpu().numpy()
        # return y_150, y_200, y_250, y
        X = self.scaler_X.transform(X)
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)

        self.SAEModel_150.eval()
        with torch.no_grad():
            y_150 = self.SAEModel_150(X, 0, PreTrain=False).cpu().numpy()
        self.SAEModel_200.eval()
        with torch.no_grad():
            y_200 = self.SAEModel_200(X, 0, PreTrain=False).cpu().numpy()
        self.SAEModel_250.eval()
        with torch.no_grad():
            y_250 = self.SAEModel_250(X, 0, PreTrain=False).cpu().numpy()
        self.StackedAutoEncoderModel.eval()
        with torch.no_grad():
            y = self.StackedAutoEncoderModel(X, 0, PreTrain=False).cpu().numpy()
        return y_150, y_200, y_250, y


def save_res(config, res):
    with open(os.path.join(config.res_dir_path, 'sup_para.txt'), encoding="utf-8-sig", mode="a") as file:
        file.write('\n')
        file.write('\nmae：{}'.format(res['mae']))
        file.write('\nmape：{}'.format(res['mape']))
        file.write('\nmse：{}'.format(res['mse']))
        file.write('\nrmse：{}'.format(res['rmse']))
        file.write('\nr2：{}'.format(res['r2']))
        file.write('\n命中率2.5%：{}'.format(res['shot2_5']))
        file.write('\n命中率5%：{}'.format(res['shot5']))
        file.write('\n命中率10%：{}'.format(res['shot10']))


def main(config):
    # ******************************************************************
    data_path = config.data_path
    shuffle = config.shuffle
    add_noise = config.add_noise
    smooth_data = config.smooth_data
    bin_width = config.bin_width
    window_size = config.window_size
    poly_order = config.poly_order
    y_index = config.y_index
    show_plot = config.show_plot
    train = config.train
    # ******************************************************************
    var_num = DICT_Y['K+'] + 1
    df = pd.read_csv(data_path, encoding='utf-8-sig')
    data = np.array(df[VAR_LIST[0:-1]])

    for i in range(data.shape[0]):
        img_name = data[i][0]  # 2022_5_27_16_0_30_27.png
        splited = img_name.split('.')[:-1][0].split('_')  # ['2022', '5', '27', '16', '0', '30', '27']
        for j in range(len(splited)):  # [2022, 5, 27, 16, 0, 30, 27]
            splited[j] = int(splited[j])
        time = datetime(*splited)  # 2022-05-27 16:00:30.000027
        data[i][0] = time

    if smooth_data:
        # 整体平滑
        data[:, 1:var_num] = savgol_filter_A(data[:, 1:var_num], window_size=window_size, poly_order=poly_order)

    labeled_index = []
    unlabeled_index = []
    for i in range(data.shape[0]):
        if data[i][y_index] == -1:
            unlabeled_index.append(i)
        else:
            labeled_index.append(i)
    labeled_data = data[labeled_index, :]
    unlabeled_data = data[unlabeled_index, :]

    if True:
        # 顺序划分
        dataset, data_size = split_by_order(labeled_data[:, :var_num], labeled_data[:, y_index], shuffle=shuffle)
        # 按区间划分数据集
        # dataset, data_size = split_by_bin(labeled_data[:, :var_num], labeled_data[:, y_index],
        #                                   bin_width=bin_width, inside_shuffle=shuffle)
        train_data, val_data, test_data = dataset
        train_num, val_num, test_num = data_size

    if train:
        train_labeled_X = train_data[:, :var_num]
        train_labeled_y = train_data[:, -1].reshape(-1, 1)

        val_X = val_data[:, 1:var_num]
        val_y = val_data[:, -1].reshape(-1, 1)
    else:
        train_labeled_X = np.append(train_data[:, :var_num], val_data[:, :var_num], axis=0)
        train_labeled_y = np.append(train_data[:, -1].reshape(-1, 1), val_data[:, -1].reshape(-1, 1), axis=0)

        val_X = test_data[:, 1:var_num]
        val_y = test_data[:, -1].reshape(-1, 1)

    train_unlabeled_X = unlabeled_data[:, :var_num]

    scaler_y = MinMaxScaler()
    train_labeled_y = scaler_y.fit_transform(train_labeled_y)
    val_y = scaler_y.transform(val_y)

    if add_noise:
        # 按照与prototy的距离加噪
        train_labeled_X[:, 1:], train_labeled_y = cos_sim_noise(train_labeled_X[:, 1:], train_labeled_y, bias_factor=2)

    print('训练集有标签：', train_labeled_X.shape)
    print('训练集无标签：', train_unlabeled_X.shape)
    if train:
        print('验证集：', val_X.shape)
    else:
        print('测试集：', val_X.shape)

    mdl = SAEModel(config=config).fit(train_labeled_X, train_labeled_y, train_unlabeled_X)

    # train_labeled_y = scaler_y.inverse_transform(train_labeled_y)
    # pred_train_y = mdl.predict(train_labeled_X[:, 1:])
    # pred_train_y = scaler_y.inverse_transform(pred_train_y)
    # train_res = cal_metrics(y_true=train_labeled_y, y_pred=pred_train_y)
    # val_y = scaler_y.inverse_transform(val_y)
    pre_test_y = mdl.predict(val_X)
    # --------------------------------------------------------------------
    for i in range(4):
        # temp_pre_test_y = scaler_y.inverse_transform(pre_test_y[i])
        temp_pre_test_y = pre_test_y[i]
        val_res = cal_metrics(y_true=val_y, y_pred=temp_pre_test_y)

        # save_res(config, train_res)
        save_res(config, val_res)

        if show_plot:
            # draw_pred_curve(train_labeled_y, pred_train_y, title='train_set', fig_size=(14, 10),
            #                 save_path=os.path.join(config.res_dir_path, '训练集结果.png'))
            draw_pred_curve(val_y, temp_pre_test_y, title='test_set', fig_size=(14, 10),
                            save_path=os.path.join(config.res_dir_path, '测试集结果_e{}.png'.format(150 + i * 50)))

        res_list = [config.time, 150 + i * 50, config.unsp_epoch, config.sp_batch_size, config.unsp_batch_size,
                    config.sp_lr, config.unsp_lr,
                    config.K_t, config.K_s, config.delta_t, config.delta_s, config.snp_loss_weight]
        for k, v in val_res.items():
            res_list.append(v)
        with open(config.res_save_path, mode='a', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(res_list)

        if i == 3:
            return val_res
    # --------------------------------------------------------------------
    # pred_train_y = scaler_y.inverse_transform(pred_train_y)
    # train_res = cal_metrics(y_true=train_labeled_y, y_pred=pred_train_y)
    #
    # val_y = scaler_y.inverse_transform(val_y)
    # pre_test_y = mdl.predict(val_X)
    # pre_test_y = scaler_y.inverse_transform(pre_test_y)
    # val_res = cal_metrics(y_true=val_y, y_pred=pre_test_y)
    #
    # save_res(config, train_res)
    # save_res(config, val_res)
    #
    # if show_plot:
    #     draw_pred_curve(train_labeled_y, pred_train_y, title='train_set', fig_size=(14, 10),
    #                     save_path=os.path.join(config.res_dir_path, '训练集结果.png'))
    #     draw_pred_curve(val_y, pre_test_y, title='test_set', fig_size=(14, 10),
    #                     save_path=os.path.join(config.res_dir_path, '测试集结果.png'))
    #
    # res_list = [config.time, config.sup_epoch, config.unsp_epoch, config.sp_batch_size, config.unsp_batch_size,
    #             config.sp_lr, config.unsp_lr,
    #             config.K_t, config.K_s, config.delta_t, config.delta_s, config.snp_loss_weight]
    # for k, v in val_res.items():
    #     res_list.append(v)
    # with open(config.res_save_path, mode='a', newline='', encoding='utf-8-sig') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(res_list)
    # return val_res


if __name__ == '__main__':
    config = argparse.ArgumentParser()
    base_dir_path = r'./'
    config.res_save_path = os.path.join(base_dir_path, '测试集结果.csv')
    if not os.path.exists(config.res_save_path):
        with open(config.res_save_path, mode='w', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['time', 'sup_epoch', 'unsp_epoch', 'sp_batch_size', 'unsp_batch_size', 'sp_lr', 'unsp_lr',
                             'K_t', 'K_s', 'delta_t', 'delta_s', 'snp_loss_weight',
                             'mae', 'mape', 'mse', 'rmse', 'r2', '命中率2.5%', '命中率5%', '命中率10%'])
            csvfile.close()

    config.data_path = r'../A2_data/202201/512/1月512有标签加无标签_v3.csv'
    config.shuffle = False
    config.add_noise = False
    config.smooth_data = True
    config.bin_width = 1
    config.window_size = 31
    config.poly_order = 5
    config.y_index = DICT_Y['K+'] + 1
    config.save_plot = False
    config.show_plot = True
    config.device = 'cpu'
    config.seed = 1024

    config.AEList = [25, 23, 18, 14]
    config.sup_epoch = 300
    config.unsp_epoch = 35
    config.sp_batch_size = 128
    config.unsp_batch_size = 32
    config.sp_lr = 0.001
    config.unsp_lr = 0.01

    config.K_t = 5
    config.delta_t = 0.2
    config.K_s = 5
    config.delta_s = 0.8

    config.train = False
    config.unlab_ratio = 1
    config.unlab_ratio = int(1. / config.unlab_ratio)

    # ---------------------------------
    snp_loss_weight = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    para_list = [snp_loss_weight]
    # ---------------------------------
    best_r2 = -999
    best_rmse = 999
    best_para = None

    for para in itertools.product(*para_list):
        config.snp_loss_weight = 0.5
        config.tnp_loss_weight = 1 - config.snp_loss_weight

        config.save_freq = int(config.sup_epoch / 2)
        t1 = realtime.time()
        print('-'*50)
        print('snp:{},tnp:{}'.format(config.snp_loss_weight,config.tnp_loss_weight))
        config.time = '{}_{}_{}_{}_{}_{}'.format(datetime.now().year, datetime.now().month, datetime.now().day,
                                                 datetime.now().hour, datetime.now().minute, datetime.now().microsecond)
        config.res_dir_path = os.path.join(base_dir_path, '{}'.format(config.time))

        val_res = main(config)
        t2 = realtime.time()
        print('耗时：{}分钟'.format((t2 - t1) / 60))

        if val_res['rmse'] <= best_rmse and val_res['r2'] >= best_r2:
            best_para = para
            best_rmse = val_res['rmse']
            best_r2 = val_res['r2']
        print('当前最优参数：{}'.format(best_para))
        break