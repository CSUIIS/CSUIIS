#!coding:utf-8

from sklearn.preprocessing import MinMaxScaler
import pandas as pd

from prediction.utils.CONST_VAR import DICT_Y, VAR_LIST
from prediction.utils.smooth_data import savgol_filter_A
# from prediction.MT_TSNLNet.architectures.model_reg import TestNet
from prediction.MT_TSNLNet.model import MTTSNLNet
from prediction.NN.dataset import *

from prediction.utils.split_dataset import *
from prediction.utils.metrics import cal_metrics, cal_metrics_avg
from prediction.utils.visualization import draw_pred_curve, Plot

# ---------------------------------------
data_path = r'../A2_data/202201/512/1月512有标签加无标签_v3.csv'
y_index = DICT_Y['K+']
shuffle = False
LABELED_TRAIN_RATIO = 0.8
bin_width = 1
smooth_data = False
window_size = 31
poly_order = 5
# ---------------------------------------
var_num = DICT_Y['K+']
df = pd.read_csv(data_path, encoding='utf-8-sig')
total_data = np.array(df[VAR_LIST[1:-1]])

if smooth_data:
    total_data[:, :var_num] = savgol_filter_A(total_data[:, :var_num], window_size, poly_order)

labeled_index = []
unlabeled_index = []
for i in range(total_data.shape[0]):
    if total_data[i][y_index] == -1:
        unlabeled_index.append(i)
    else:
        labeled_index.append(i)
labeled_data = total_data[labeled_index, :]
unlabeled_data = total_data[unlabeled_index, :]

# 按区间划分数据集
if True:
    # 顺序划分
    dataset, data_size = split_by_order(labeled_data[:, :var_num], labeled_data[:, y_index],
                                        shuffle=shuffle, y_shuffle=False)
    # dataset, data_size = split_by_bin(labeled_data[:, :var_num], labeled_data[:, y_index],
    #                                   train_ratio=LABELED_TRAIN_RATIO,
    #                                   bin_width=bin_width, inside_shuffle=shuffle, seed=1024)
    train_data, val_data, test_data = dataset
    train_num, val_num, test_num = data_size

train_labelled_X = np.append(train_data[:, :var_num], val_data[:, :var_num], axis=0)
train_labelled_y = np.append(train_data[:, -1].reshape(-1, 1), val_data[:, -1].reshape(-1, 1), axis=0)

# 计算训练集结果
# validation_X = train_data[:, :var_num]
# validation_y = train_data[:, -1].reshape(-1, 1)
# 计算验证集结果
# validation_X = val_data[:, :var_num]
# validation_y = val_data[:, -1].reshape(-1, 1)
# 计算测试集结果
validation_X = test_data[:, :var_num]
validation_y = test_data[:, -1].reshape(-1, 1)

train_unlabelled_X = unlabeled_data[:, 0:var_num]
train_unlabelled_y = unlabeled_data[:, y_index].reshape(-1)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

train_X = np.append(train_labelled_X, train_unlabelled_X, axis=0)
train_X = scaler_X.fit_transform(train_X)

train_labelled_y = scaler_y.fit_transform(train_labelled_y)

train_y = np.append(train_labelled_y, train_unlabelled_y.reshape(-1, 1), axis=0)

# # -------------------------------------------
# df_val = pd.read_csv(r'D:\Desktop\202303_粗选512_按时间排序.csv', encoding='utf-8-sig')
# val_data = np.array(df_val[VAR_LIST[1:-1]])
# if smooth_data:
#     val_data[:, :var_num] = savgol_filter_A(val_data[:, :var_num], window_size, poly_order)
# validation_X = val_data[:,:25]
# validation_y = val_data[:,y_index].reshape(-1,1)
# # -------------------------------------------

validation_X = scaler_X.transform(validation_X)

print('训练集有标签：', train_labelled_X.shape)
print('训练集无标签：', train_unlabelled_X.shape)
print('训练集：', train_X.shape)
print('测试集：', validation_X.shape)
print(validation_X.shape)
print(validation_y.shape)

net = MTTSNLNet([25, 23, 18, 14])
model_weight_path = r'model_epoch_299.pth'
net.load_state_dict(torch.load(model_weight_path, map_location='cpu')['weight_s'])

with torch.no_grad():
    # # 计算时间
    # net.eval()
    # validation_X = torch.tensor(validation_X, dtype=torch.float)[0].reshape(1,-1)
    # import time
    # a = time.clock()
    # for i in range(950):
    #     pred_y, _ = net(validation_X)
    # b = time.clock()
    # print(b - a)
    # exit(0)

    # 测试集结果
    net.eval()
    validation_X = torch.tensor(validation_X, dtype=torch.float)
    pred_y, _ = net(validation_X)
    pred_y = pred_y.numpy().reshape(-1, 1)
    validation_y = scaler_y.transform(validation_y)
    cal_metrics(validation_y, pred_y)
    pred_y = scaler_y.inverse_transform(pred_y)
    validation_y = scaler_y.inverse_transform(validation_y)
    _, y_true, y_pred = cal_metrics_avg(validation_y, pred_y)
    Plot(y_true=y_true,y_pre=y_pred)
    draw_pred_curve(validation_y,pred_y,fig_size=(14, 10),title='',
                    save_path=None)

    # validation_y = scaler_y.transform(validation_y)
    # pred_y = scaler_y.transform(pred_y)
    # res = np.append(validation_y,pred_y,axis=1)
    # df = pd.DataFrame(res,columns=['真实值','预测值'])
    # df.to_csv(r'D:\Desktop\论文1\预测结果\NN+MT+STNP.csv',encoding='utf-8-sig')