#!coding:utf-8
import csv
import itertools
import os
import time
from datetime import datetime

import torch
from sklearn.preprocessing import MinMaxScaler
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset, Dataset
import pandas as pd

from prediction.MT_TSNLNet import trainer
from prediction.MT_TSNLNet.model import MTTSNLNet
from prediction.MT_TSNLNet.utils.ramps import exp_warmup
from prediction.MT_TSNLNet.utils.config import parse_commandline_args
from prediction.MT_TSNLNet.utils.data_utils import DataSetWarpper
from prediction.MT_TSNLNet.utils.data_utils import TwoStreamBatchSampler
from prediction.MT_TSNLNet.utils.data_utils import TransformTwice as twice

from prediction.MT_TSNLNet.trainer import *
from prediction.utils.CONST_VAR import DICT_Y, VAR_LIST
from prediction.utils.noise import *
from prediction.utils.smooth_data import savgol_filter_A
from prediction.utils.split_dataset import *


class MyDataset(Dataset):
    # Initialization
    def __init__(self, data, label):
        self.data, self.label = data, label
        self.transform = None
        self.num_classes = 1

    # Get item
    def __getitem__(self, index):
        return (self.data[index, :], torch.tensor(index, dtype=torch.long)), self.label[index, :],

    # Get length
    def __len__(self):
        return self.data.shape[0]


build_model = {
    'mt-tsnl': trainer.Trainer,
}


def create_loaders_regv1(trainset, evalset, label_idxs, unlab_idxs, config):
    # use two data stream
    if config.data_twice:
        trainset.transform = twice(trainset.transform)
    # enable indexs of samples
    if config.data_idxs:
        trainset = DataSetWarpper(trainset, 1)
    ## two-stream batch loader
    batch_size = config.sup_batch_size + config.usp_batch_size
    batch_sampler = TwoStreamBatchSampler(
        unlab_idxs, label_idxs, batch_size, config.sup_batch_size)

    train_loader = torch.utils.data.DataLoader(trainset,
                                               batch_sampler=batch_sampler,
                                               num_workers=config.workers,
                                               pin_memory=True)
    ## test batch loader
    eval_loader = torch.utils.data.DataLoader(evalset,
                                              batch_size=len(trainset),
                                              shuffle=False,
                                              num_workers=2 * config.workers,
                                              pin_memory=True,
                                              drop_last=False)
    return train_loader, eval_loader


def create_optim(params, config):
    if config.optim == 'sgd':
        optimizer = optim.SGD(params, config.lr,
                              momentum=config.momentum,
                              weight_decay=config.weight_decay,
                              nesterov=config.nesterov)
    elif config.optim == 'adam':
        optimizer = optim.Adam(params, config.lr)
    return optimizer


def create_lr_scheduler(optimizer, config):
    if config.lr_scheduler == 'cos':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max=config.epochs,
                                                   eta_min=config.min_lr)
    elif config.lr_scheduler == 'multistep':
        if config.steps is None: return None
        if isinstance(config.steps, int): config.steps = [config.steps]
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             milestones=config.steps,
                                             gamma=config.gamma)
    elif config.lr_scheduler == 'exp-warmup':
        lr_lambda = exp_warmup(config.rampup_length,
                               config.rampdown_length,
                               config.epochs)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif config.lr_scheduler == 'none':
        scheduler = None
    else:
        raise ValueError("No such scheduler: {}".format(config.lr_scheduler))
    return scheduler


def create_dataset_reg_v1(config):
    df = pd.read_csv(config.csv_data_path, encoding='utf-8-sig')

    var_num = 26
    total_data = np.array(df[VAR_LIST[0:-1]])

    for i in range(total_data.shape[0]):
        img_name = total_data[i][0]  # 2022_5_27_16_0_30_27.png
        splited = img_name.split('.')[:-1][0].split('_')  # ['2022', '5', '27', '16', '0', '30', '27']
        for j in range(len(splited)):  # [2022, 5, 27, 16, 0, 30, 27]
            splited[j] = int(splited[j])
        time = datetime(*splited)  # 2022-05-27 16:00:30.000027
        total_data[i][0] = time

    if config.smooth_data:
        total_data[:, 1:var_num] = savgol_filter_A(total_data[:, 1:var_num], config.window_size, config.poly_order)

    labeled_index = []
    unlabeled_index = []
    for i in range(total_data.shape[0]):
        if total_data[i][config.y_index] == -1:
            unlabeled_index.append(i)
        else:
            labeled_index.append(i)
    labeled_data = total_data[labeled_index, :]
    unlabeled_data = total_data[unlabeled_index, :]

    # 按区间划分数据集
    if config.split_type == 1:
        dataset, data_size = split_by_bin(labeled_data[:, :var_num], labeled_data[:, config.y_index],
                                          train_ratio=config.LABELED_TRAIN_RATIO,
                                          bin_width=config.bin_width, inside_shuffle=config.shuffle, seed=1024)
    # 顺序划分
    else:
        dataset, data_size = split_by_order(labeled_data[:, :var_num], labeled_data[:, config.y_index],train_ratio=config.LABELED_TRAIN_RATIO,
                                            shuffle=config.shuffle)
    train_data, val_data, test_data = dataset
    train_num, val_num, test_num = data_size

    if config.train:
        train_labelled_X = train_data[:, :var_num]
        train_labelled_y = train_data[:, -1].reshape(-1, 1)

        validation_X = val_data[:, 1:var_num]
        validation_y = val_data[:, -1].reshape(-1, 1)
    else:
        train_labelled_X = np.append(train_data[:, :var_num], val_data[:, :var_num], axis=0)
        train_labelled_y = np.append(train_data[:, -1].reshape(-1, 1), val_data[:, -1].reshape(-1, 1), axis=0)

        validation_X = test_data[:, 1:var_num]
        validation_y = test_data[:, -1].reshape(-1, 1)

    train_unlabelled_X = unlabeled_data[:, 0:var_num]
    train_unlabelled_y = unlabeled_data[:, config.y_index].reshape(-1)

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    train_X = np.append(train_labelled_X, train_unlabelled_X, axis=0)
    train_X[:, 1:] = scaler_X.fit_transform(train_X[:, 1:])

    train_labelled_y = scaler_y.fit_transform(train_labelled_y)

    if config.add_noise:
        # 按照与prototy的距离加噪
        train_X[0:train_num, 1:], train_labeled_y = cos_sim_noise(train_X[0:train_num, 1:],
                                                                  train_labelled_y, bias_factor=2)

    train_y = np.append(train_labelled_y, train_unlabelled_y.reshape(-1, 1), axis=0)

    validation_X = scaler_X.transform(validation_X)
    validation_y = scaler_y.transform(validation_y)

    print('训练集有标签：', train_labelled_X.shape)
    print('训练集无标签：', train_unlabelled_X.shape)
    print('训练集：', train_X.shape)
    if config.train:
        print('验证集：', validation_X.shape)
    else:
        print('测试集：', validation_X.shape)

    label_idxs = []
    unlab_idxs = []
    for i in range(train_y.shape[0]):
        if train_y[i] == -1:
            unlab_idxs.append(i)
        else:
            label_idxs.append(i)

    # Create DataLoaders
    trainset = MyDataset(torch.FloatTensor(train_X[:, 1:].astype(float)), torch.FloatTensor(train_y.astype(float)))
    evalset = TensorDataset(torch.FloatTensor(validation_X), torch.FloatTensor(validation_y))
    return trainset, evalset, label_idxs, unlab_idxs, scaler_y, train_X


def run_reg(config):
    print(config)
    print("pytorch version : {}".format(torch.__version__))
    ## create save directory
    if config.save_freq != 0 and not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
    ## prepare data
    trainset, evalset, label_idxs, unlab_idxs, scaler_y, train_X = create_dataset_reg_v1(config)

    loaders = create_loaders_regv1(trainset, evalset, label_idxs, unlab_idxs, config=config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net = MTTSNLNet(config.net_struct, dropout=config.dropout)
    net = net.to(device)
    optimizer = create_optim(net.parameters(), config)
    scheduler = create_lr_scheduler(optimizer, config)

    # run the model
    net2 = MTTSNLNet(config.net_struct, dropout=config.dropout)
    net2 = net2.to(device)
    trainer = build_model[config.model](net, net2, optimizer, device, scaler_y, config, train_X)

    # MTbased = set(['mt', 'ict'])
    # if config.model[-4:-2] in MTbased or config.model[-5:] == 'match':
    #
    # else:
    #     trainer = build_model[config.model](net, optimizer, device, scaler_y, config, train_X)
    rmse, r2 = trainer.loop(config.epochs, *loaders, scheduler=scheduler)
    return rmse, r2


if __name__ == '__main__':
    config = parse_commandline_args()
    config.optim = 'sgd'
    config.momentum = 0.9
    config.weight_decay = 5e-4
    config.nesterov = True
    config.rampup_length = 80
    config.rampdown_length = 50
    config.workers = 0

    config.shuffle = False
    config.add_noise = False
    config.smooth_data = False
    config.bin_width = 1
    config.window_size = 31
    config.poly_order = 5
    config.y_index = DICT_Y['K+'] + 1
    config.cuda = 'cpu'

    config.seed = 1024
    if config.seed is not None:
        torch.manual_seed(config.seed)
    config.arch = 'mlp'
    config.print_freq = 200
    config.train = False

    config.save_dir = r'./'
    config.res_csv_path = os.path.join(config.save_dir, '测试集结果.csv')
    if not os.path.exists(config.res_csv_path):
        with open(config.res_csv_path, mode='w', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['mae', 'mape', 'mse', 'rmse', 'r2', '命中率2.5%', '命中率5%', '命中率10%',
                             'time', 'sup_batch_size', 'usp_batch_size', 'lr', 'net_struct', 'dropout', 'model_type',
                             'num_labels', 'model', 'epochs',
                             'cons_loss_weight', 'recon_loss_weight', 'ema_decay',
                             'lr_scheduler', 'min_lr',
                             'K_t', 'delta_t', 'K_s', 'delta_s', 'snp_loss_weight'])
            csvfile.close()

    config.img_size = 512
    config.csv_data_path = r'../A2_data/202201/512/1月512有标签加无标签_v3.csv'
    config.fusion = False

    config.sup_batch_size = 128
    config.usp_batch_size = 128
    config.split_type = 2
    if config.split_type == 1:
        config.num_labels = 7569  # 按区间划分数据集，训练集0.6
    else:
        config.num_labels = 7478  # 1月512有标签训练集 (0.58比例)
    config.model = 'mt-tsnl'
    config.cons_loss_weight = 0.3
    config.recon_loss_weight = 0.3
    config.ema_decay = 0.99
    config.epochs = 10
    config.lr = 0.001
    config.lr_scheduler = 'cos'
    config.min_lr = 1e-5
    config.save_freq = config.epochs / 2

    config.LABELED_TRAIN_RATIO = 0.6

    config.net_struct = [25, 23, 18, 14]
    config.dropout = None

    config.K_s = 5
    config.delta_s = 0.5
    config.K_t = 4
    config.delta_t = 0.2

    config.snp_loss_weight = 0.5
    config.tnp_loss_weight = 1 - config.snp_loss_weight
    # ---------------------------------
    K_s = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    delta_s = [0.1, 0.2, 0.5, 0.8, 1, 2, 5]
    para_list = [K_s, delta_s]
    # ---------------------------------

    best_r2 = -999
    best_rmse = 999
    best_para = None

    for para in itertools.product(*para_list):
        config.time = '{}_{}_{}_{}_{}_{}'.format(datetime.now().year, datetime.now().month, datetime.now().day,
                                                 datetime.now().hour, datetime.now().minute, datetime.now().microsecond)
        t1 = time.time()
        print('当前参数：{}'.format(para))

        config.K_s = para[0]
        config.delta_s = para[1]

        rmse, r2 = run_reg(config)
        t2 = time.time()
        print(t2 - t1)
        if rmse <= best_rmse and r2 >= best_r2:
            best_para = para
            best_rmse = rmse
            best_r2 = r2
        print('当前最优参数：{}'.format(best_para))
