import numpy as np
import pandas as pd
import scipy.signal
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import MinMaxScaler
from prediction.utils.CONST_VAR import VAR_LIST, DICT_Y
from prediction.utils.metrics import cal_metrics
from prediction.utils.noise import *
from prediction.utils.smooth_data import savgol_filter_A
from prediction.utils.split_dataset import *
from prediction.utils.visualization import draw_pred_curve


def cosine_similarity(x, y):
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    # return 0.5 * (num / denom) + 0.5
    return num / denom


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
    def __init__(self, data, label, mode='2D'):
        self.data, self.label, self.mode = data, label, mode

    # Get item
    def __getitem__(self, index):
        if self.mode == '2D':
            return self.data[index, :], self.label[index, :]
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
        self.decoder = nn.Linear(dim_H, dim_X + 1, bias=True)

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
def trainAE(model, trainloader, epochs, trainlayer, lr, ss_lambda):
    optimizer = torch.optim.Adam(model.SAE[trainlayer].parameters(), lr=lr)
    loss_func_X = nn.MSELoss()
    loss_func_y = nn.MSELoss()

    for j in range(epochs):
        sum_loss = 0
        for X, y in trainloader:
            y = y.reshape(-1)
            l_mask = y != -1

            Hidden, Hidden_reconst = model(X, trainlayer, PreTrain=True)
            pred_y = Hidden_reconst[:, -1]
            Hidden_reconst = Hidden_reconst[:, :-1]

            x_recon_loss = loss_func_X(Hidden, Hidden_reconst)
            y_recon_loss = loss_func_y(y[l_mask], pred_y[l_mask])

            loss = x_recon_loss + ss_lambda * y_recon_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += loss.detach().item()
        print('无监督预训练第{}层的第{}个epoch, '.format(trainlayer + 1, j + 1),
              ',其Loss的大小是:x_recon_loss:{}, y_recon_loss:{}'.format(x_recon_loss.data.cpu().numpy(),
                                                                y_recon_loss.data.cpu().numpy()))

    return model


# SAE训练的代码模型
class SAEModel(BaseEstimator, RegressorMixin):
    def __init__(self, AEList=[13, 10, 7, 5], sup_epoch=300, unsp_epoch=200, unsp_batch_size=64, sp_batch_size=64,
                 sp_lr=0.03,
                 unsp_lr=0.01, ss_lambda=1.6, device=torch.device('cuda:0'), seed=1024):
        super(SAEModel, self).__init__()
        torch.manual_seed(seed)

        # 参数分配
        self.AEList = AEList
        self.num_AE = len(AEList) - 1
        self.unsp_epoch = unsp_epoch
        self.sup_epoch = sup_epoch
        self.unsp_batch_size = unsp_batch_size
        self.sp_batch_size = sp_batch_size
        self.unsp_lr = unsp_lr
        self.sp_lr = sp_lr
        self.device = device
        self.seed = seed

        self.scaler_X = MinMaxScaler()
        self.ss_lambda = ss_lambda

        # SAE模型的创建
        self.StackedAutoEncoderModel = StackedAutoEncoder(size=AEList, device=device).to(device)

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

    # 数据拟合
    def fit(self, labeled_X, labeled_y, unlabeled_X):
        total_X = np.append(labeled_X, unlabeled_X, axis=0)
        unlabeled_y = np.ones((total_X.shape[0], 1)) * -1
        total_y = np.append(labeled_y, unlabeled_y, axis=0)
        total_X = self.scaler_X.fit_transform(total_X)
        labeled_X = self.scaler_X.transform(labeled_X)

        unsup_dataset = MyDataset_unsup(torch.tensor(total_X, dtype=torch.float32, device=self.device),
                                        torch.tensor(total_y, dtype=torch.float32, device=self.device), '2D')
        sup_dataset = MyDataset(torch.tensor(labeled_X, dtype=torch.float32, device=self.device),
                                torch.tensor(labeled_y, dtype=torch.float32, device=self.device), '2D')

        un_trainloader = DataLoader(unsup_dataset, batch_size=self.unsp_batch_size, shuffle=True)
        trainloader = DataLoader(sup_dataset, batch_size=self.sp_batch_size, shuffle=True)
        print(next(self.StackedAutoEncoderModel.parameters()).is_cuda)
        self.StackedAutoEncoderModel.train()

        for i in range(self.num_AE):
            print('自编码器训练第{}层:'.format(i + 1))
            self.StackedAutoEncoderModel = trainAE(model=self.StackedAutoEncoderModel, trainloader=un_trainloader,
                                                   epochs=self.unsp_epoch, trainlayer=i, lr=self.unsp_lr,
                                                   ss_lambda=self.ss_lambda)
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
            # print('有监督微调第{}轮的Loss是{}'.format(i+1, loss.A2_data.cpu().numpy()))
            print('有监督微调第{}轮的Loss是{}'.format(i + 1, sum_loss))
            Loss.append(sum_loss)
        # 绘制损失函数曲线
        plt.figure()
        plt.plot(range(len(Loss)), Loss, color='b')
        plt.show()
        # torch.save(self.StackedAutoEncoderModel.state_dict(), r'model/SAE_{}.pkl'.format(VAR_LIST[y_index + 1]))

        return self

    # 预测数据
    def predict(self, X):
        # X = self.scaler_X.transform(X)[0].reshape(1,-1)
        # X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        # self.StackedAutoEncoderModel.eval()
        # with torch.no_grad():
        #     import time
        #     a = time.clock()
        #     for i in range(950):
        #         y = self.StackedAutoEncoderModel(X, 0, PreTrain=False).cpu().numpy()
        #     b = time.clock()
        #     print(b-a)
        #     exit(0)
        # return y
        X = self.scaler_X.transform(X)
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        self.StackedAutoEncoderModel.eval()
        with torch.no_grad():
            y = self.StackedAutoEncoderModel(X, 0, PreTrain=False).cpu().numpy()
        return y

# ******************************************************************
data_path = r'../A2_data/202201/512/1月512有标签加无标签_v3.csv'
shuffle = False
add_noise = False
smooth_data = False
bin_width = 1
window_size = 31
poly_order = 5
TRAIN_RATIO = 0.8
net_strcut = [25, 23, 18, 14]
y_index = DICT_Y['K+']
save_path = None
save_plot = False
show_plot = False
train = False
device = 'cpu'

sup_epoch = 300
unsp_epoch = 35
sp_batch_size = 128
unsp_batch_size = 32
sp_lr = 0.001
unsp_lr = 0.01
ss_lambda = 1.2
# ******************************************************************
df = pd.read_csv(data_path, encoding='utf-8-sig')
data = np.array(df[VAR_LIST[1:-1]])
var_num = DICT_Y['K+']

if smooth_data:
    # 整体平滑
    data[:, :var_num] = savgol_filter_A(data[:, :var_num], window_size=window_size, poly_order=poly_order)

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
    # 按区间划分数据集
    # dataset, data_size = split_by_bin(labeled_data[:, :var_num], labeled_data[:, y_index],
    #                                   bin_width=bin_width, inside_shuffle=shuffle)
    # 顺序划分
    dataset, data_size = split_by_order(labeled_data[:,:var_num],labeled_data[:,y_index])
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

    val_X = test_data[:, :var_num]
    val_y = test_data[:, -1].reshape(-1, 1)

train_unlabeled_X = unlabeled_data[:, :var_num]

scaler_y = MinMaxScaler()
train_labeled_y = scaler_y.fit_transform(train_labeled_y)
val_y = scaler_y.transform(val_y)

if add_noise:
    # 按照与prototy的距离加噪
    train_labeled_X, train_labeled_y = cos_sim_noise(train_labeled_X, train_labeled_y, bias_factor=2)

print('训练集有标签：', train_labeled_X.shape)
print('训练集无标签：', train_unlabeled_X.shape)
if train:
    print('验证集：', val_X.shape)
else:
    print('测试集：', val_X.shape)

mdl = SAEModel(AEList=net_strcut, sup_epoch=sup_epoch, unsp_epoch=unsp_epoch, unsp_batch_size=unsp_batch_size,
               sp_batch_size=sp_batch_size, sp_lr=sp_lr, unsp_lr=unsp_lr,
               device=torch.device(device), ss_lambda=ss_lambda,seed=1024).fit(train_labeled_X, train_labeled_y,
                                                           train_unlabeled_X)


# 训练集结果
# train_labeled_y = scaler_y.inverse_transform(train_labeled_y)
# pred_train_y = mdl.predict(train_labeled_X)
# pred_train_y = scaler_y.inverse_transform(pred_train_y)
# cal_metrics(y_true=train_labeled_y, y_pred=pred_train_y)

# 测试集结果
# val_y = scaler_y.inverse_transform(val_y)
pred_val_y = mdl.predict(val_X)
# pred_val_y = scaler_y.inverse_transform(pred_val_y)
cal_metrics(y_true=val_y, y_pred=pred_val_y)

# if show_plot:
#     draw_pred_curve(train_labeled_y, pred_train_y, title='train_set', fig_size=(14, 10), save_path=save_path)
#     draw_pred_curve(val_y, pred_val_y, title='test_set', fig_size=(14, 10), save_path=save_path)
