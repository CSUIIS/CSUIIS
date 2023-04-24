import csv
import os

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import torch


def cal_mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true))


def smape(y_true, y_pred):
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))


def shot_ratio(y_true, y_pred, percentage=0.1):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.numpy()

    total_num = y_true.shape[0]
    shot_num = 0

    bias = y_true * percentage
    up = y_true + bias
    down = y_true - bias
    for i in range(total_num):
        pred_value = y_pred[i][0]
        # print('上界:{},下界:{}，预测值:{}'.format(up[i][0],down[i][0],pred_value))
        if pred_value > down[i][0] and pred_value < up[i][0]:
            shot_num += 1
    return shot_num / total_num


# y_true,y_pred 形状 (n,1)
def cal_metrics(y_true, y_pred, print_info=True):
    if len(y_true.shape) != 2:
        y_true.reshape(-1, 1)
    if len(y_pred.shape) != 2:
        y_pred.reshape(-1, 1)

    res = {}
    res['mae'] = mean_absolute_error(y_true=y_true, y_pred=y_pred)
    res['mape'] = cal_mape(y_true, y_pred)
    res['mse'] = mean_squared_error(y_true=y_true, y_pred=y_pred)
    res['rmse'] = np.sqrt(res['mse'])
    res['r2'] = r2_score(y_true=y_true, y_pred=y_pred)
    res['shot2_5'] = shot_ratio(y_true, y_pred, percentage=0.025)
    res['shot5'] = shot_ratio(y_true, y_pred, percentage=0.05)
    res['shot10'] = shot_ratio(y_true, y_pred, percentage=0.1)
    if print_info:
        print('------------评价指标------------')
        print('mae = {}'.format(round(res['mae'], 5)))
        print('mape = {}'.format(round(res['mape'], 5)))
        print('mse = {}'.format(round(res['mse'], 5)))
        print('rmse = {}'.format(round(res['rmse'], 5)))
        print('r2 = {}'.format(round(res['r2'], 5)))
        print('命中率2.5% = {}'.format(round(res['shot2_5'], 5)))
        print('命中率5% = {}'.format(round(res['shot5'], 5)))
        print('命中率10% = {}'.format(round(res['shot10'], 5)))
        print('------------------------------')
    return res


def cal_metrics_avg(y_true, y_pred, print_info=True):
    if len(y_true.shape) != 2:
        y_true.reshape(-1, 1)
    if len(y_pred.shape) != 2:
        y_pred.reshape(-1, 1)

    y_true_avg = []
    y_pred_avg = []
    start = 0
    for i in range(y_true.shape[0] - 1):
        if y_true[i][0] != y_true[i + 1][0] or i == y_true.shape[0] - 2:
            y_true_avg.append(np.mean(y_true[start: i + 1, :]))
            y_pred_avg.append(np.mean(y_pred[start: i + 1, :]))
            start = i + 1
    y_true = np.array(y_true_avg).reshape(-1, 1)
    y_pred = np.array(y_pred_avg).reshape(-1, 1)

    res = {}
    res['mae'] = mean_absolute_error(y_true=y_true, y_pred=y_pred)
    res['mape'] = cal_mape(y_true, y_pred)
    res['mse'] = mean_squared_error(y_true=y_true, y_pred=y_pred)
    res['rmse'] = np.sqrt(res['mse'])
    res['r2'] = r2_score(y_true=y_true, y_pred=y_pred)
    res['shot2_5'] = shot_ratio(y_true, y_pred, percentage=0.025)
    res['shot5'] = shot_ratio(y_true, y_pred, percentage=0.05)
    res['shot10'] = shot_ratio(y_true, y_pred, percentage=0.1)
    if print_info:
        print('---------[取平均]评价指标---------')
        print('mae = {}'.format(round(res['mae'], 5)))
        print('mape = {}'.format(round(res['mape'], 5)))
        print('mse = {}'.format(round(res['mse'], 5)))
        print('rmse = {}'.format(round(res['rmse'], 5)))
        print('r2 = {}'.format(round(res['r2'], 5)))
        print('命中率2.5% = {}'.format(round(res['shot2_5'], 5)))
        print('命中率5% = {}'.format(round(res['shot5'], 5)))
        print('命中率10% = {}'.format(round(res['shot10'], 5)))
        print('------------------------------')
    return res, y_true, y_pred


def save_result(csv_path,res,description):
    res_list = [description, res['mae'], res['mape'], res['mse'], res['rmse'],
                res['r2'], res['shot2_5'], res['shot5'], res['shot10']]
    if not os.path.exists(csv_path):
        with open(csv_path, mode='w', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['描述','mae', 'mape', 'mse', 'rmse', 'r2', '命中率2.5%', '命中率5%', '命中率10%'])
            writer.writerow(res_list)
            csvfile.close()
    else:
        with open(csv_path, mode='a', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(res_list)
            csvfile.close()