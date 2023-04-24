import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
import joblib
import seaborn as sns
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import r2_score
from prediction.utils.noise import *
from prediction.utils.smooth_data import *
from prediction.utils.split_dataset import *
from prediction.utils.visualization import draw_hist, draw_pred_curve, Plot
from prediction.utils.metrics import *
from prediction.utils.CONST_VAR import *

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False

# *************************************************
data_path = '../A2_data/202201/512/1月512有标签.csv'

shuffle = False
add_noise = False
smooth_data = False
window_size = 31
poly_order = 5
bin_width = 1
sample_weight = False
y_index = DICT_Y['K+']
save_plot = False
show_plot = True
TRAIN_RATIO = 0.8
train = True
# *************************************************
var_num = DICT_Y['K+']
df = pd.read_csv(data_path, encoding='utf-8-sig')


data = np.array(df[VAR_LIST[1:-1]])

if smooth_data:
    data[:, :var_num] = savgol_filter_A(data[:, :var_num], window_size=window_size, poly_order=poly_order)

if True:
    # 顺序划分
    dataset, data_size = split_by_order(data[:, :var_num], data[:, y_index], train_ratio=TRAIN_RATIO,
                                        shuffle=False, seed=None)
    # 按区间划分数据集
    # dataset, data_size = split_by_bin(data[:, :var_num], data[:, y_index], train_ratio=0.8,
    #                                   bin_width=bin_width, inside_shuffle=shuffle, seed=1024)
    # 按标签划分
    # dataset, data_size, _ = split_by_label(data[:, :var_num], data[:, y_index:y_index + 5])
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

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

train_X = scaler_X.fit_transform(train_X)
train_y = scaler_y.fit_transform(train_y)
test_X = scaler_X.transform(val_X)
test_y = scaler_y.transform(val_y)
print(train_X.shape)
print(train_y.shape)
print(test_X.shape)
print(test_y.shape)

if add_noise:
    # 按照与prototy的距离加噪
    train_X, train_y = cos_sim_noise(train_X, train_y, bias_factor=2)
    # 高斯噪声
    # train_y = gauss_noise(train_y,mu=0,sigma=sigma_,seed=None)
    # 随机噪声
    # train_y = random_noise(train_y,n_range=0.1)

weight = []
for i in range(train_y.shape[0]):
    down_bound = int(train_y[i][0])
    up_bound = int(train_y[i][0]) + 1
    temp = train_y[train_y >= down_bound]
    temp = temp[temp <= up_bound]
    ratio = temp.shape[0] / train_y.shape[0]
    weight.append(1 / ratio)
# print(weight)
weight = np.array(weight).reshape((-1, 1))
weight = MinMaxScaler().fit_transform(weight)
weight = list(weight.reshape((-1)) + 0.25)
# print(weight)

# model = SVR(C=1,kernel='rbf')
model = RandomForestRegressor(random_state=1024, n_estimators=100,max_leaf_nodes=76)
# model = AdaBoostRegressor(n_estimators=5)
if sample_weight:
    performance = model.fit(train_X, train_y.flatten(), sample_weight=weight)
else:
    performance = model.fit(train_X, train_y.flatten())
pred_y = model.predict(test_X)

train_r2 = performance.score(train_X, train_y)
test_r2 = performance.score(test_X, test_y)
print("训练集合上R^2 = {:.3f}".format(train_r2))
print("测试集合上R^2 = {:.3f} ".format(test_r2))

pred_y = pred_y.reshape(-1, 1)
# 反归一化
test_y = scaler_y.inverse_transform(test_y)
pred_y = scaler_y.inverse_transform(pred_y)

res = cal_metrics(test_y, pred_y)

if show_plot:
    draw_pred_curve(test_y, pred_y, fig_size=(14, 10))