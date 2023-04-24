from prediction.utils.CONST_VAR import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.decomposition import PCA

from prediction.utils.smooth_data import savgol_filter_A

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False


def t_sne(data_path):
    # *************************************************
    # *************************************************

    df = pd.read_csv(data_path)
    data = np.array(df[VAR_LIST[1:-6]])
    data = savgol_filter_A(data, window_size=15, poly_order=7)
    print(data.shape)

    tsne = TSNE()
    tsne.fit_transform(data[:, 0:25])  # 进行数据降维

    # 降维成2维，加入到dataframe中
    df['x1'] = tsne.embedding_[:, 0]
    df['y1'] = tsne.embedding_[:, 1]
    plt.figure(figsize=(10,6))
    sns.scatterplot(x='x1', y='y1', hue='class', data=df)
    plt.show()

# t_sne(r'D:\Desktop\1.csv')


def feature_importance_rf(model, save_path, save_plot=False):
    var_list_copy = var_list[1:19].copy()
    sorted_idx = model.feature_importances_.argsort()
    plt.barh(list(np.array(var_list_copy)[sorted_idx]), model.feature_importances_[sorted_idx])
    plt.xlabel("Random Forest Feature Importance")
    if save_plot:
        plt.savefig(save_path)
    plt.show()


# 各个品位区间的统计量和其他区间余弦相似度柱状图


# 绘制柱状图   传入numpy数组形状 (n,)    width 区间大小
def draw_hist(array, width, range, save_path=None):
    # plt.hist(array,range=(19,31))
    sns.histplot(array, binwidth=width, binrange=range)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def pca(data_path):
    # ***********************************************************************
    df = pd.read_csv(data_path)
    # ***********************************************************************

    total_data = np.array(df[VAR_LIST[1:-6]])
    # total_data[:,0:25] = savgol_filter_A(total_data[:,0:25], window_size=15, poly_order=7)
    pca = PCA(n_components=2)
    reduced_x = pca.fit_transform(total_data[:, 0:25])  # 得到了pca降到2维的数据

    df['x'] = reduced_x[:, 0]
    df['y'] = reduced_x[:, 1]
    plt.figure(figsize=(10,6))
    sns.scatterplot(x='x', y='y', hue='class', data=df)
    plt.show()
# pca(r'D:\Desktop\1.csv')


def heatmap(df):
    plt.figure(figsize=(20, 20))
    sns.heatmap(df[var_list[1:-1]].corr(), annot=True)
    # plt.savefig('heatmap.png')
    plt.show()


# dataframe绘图
def draw_df(df, save_path=None):
    df.plot(subplots=True, figsize=(25, 25))
    if save_path:
        plt.savefig(save_path)
    plt.show()


def draw_pred_curve(real_y, pred_y, fig_size=None, title=None, save_path=None):
    samplt_n = len(pred_y)
    if fig_size is not None:
        plt.figure(figsize=fig_size)
    else:
        plt.figure()
    plt.plot(range(samplt_n), pred_y, color='b', label='y_pred')
    plt.plot(range(samplt_n), real_y, color='r', label='y_true')
    plt.legend()
    if title is not None:
        plt.title(title)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def Plot(y_pre, y_true, title=None, save_path=None):
    # mode : train or test ( equal Training or Testing)
    plt.figure(figsize=(10, 5), dpi=200)  # 图的大小
    # 绘制预测值的曲线，x y轴以及名称的设置
    plt.plot(range(len(y_pre)), y_pre, color='r', label='y_pre', marker='D', linewidth=2.0, markersize=5)
    # 绘制真实值的曲线，x y轴以及名称的设置
    plt.plot(range(len(y_pre)), y_true, color='k', label='y_true', marker='o', linewidth=2.0, markersize=5)
    # legend给图加上图例
    plt.legend(fontsize=15, facecolor='gainsboro')
    if title:
        plt.title(title, fontsize=20)
    plt.xlabel('Sample Number', fontsize=20)
    plt.ylabel('Output Value', fontsize=20)
    plt.xticks(fontsize=15)  # 设置字体大小
    plt.yticks(fontsize=15)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def Abs_error_curve(y_pre, y_true):
    abs_error = np.abs(y_true - y_pre)
    percent_error = abs_error / np.abs(y_true)
    # print(percent_error)
    Samples = len(y_pre)
    sum_num = 0
    for i in range(Samples):
        if (percent_error[i] < 0.5):
            sum_num = sum_num + 1
    Percent = sum_num / Samples
    print(Percent)
    plt.figure(figsize=(10, 5), dpi=200)
    plt.plot(range(len(y_pre)), abs_error, color='r', label='y_pre', marker='D', linewidth=2.0, markersize=5)
    plt.legend(fontsize=15, facecolor='gainsboro')
    plt.title('Abs_error Curve', fontsize=20)
    plt.xlabel('Sample Number', fontsize=20)
    plt.ylabel('Error Value', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()


def data_distribution(csv_data_path, train_ratio, bin_width=1):
    from prediction.utils.split_dataset import split_by_bin
    var_num = DICT_Y['K+']
    y_index = DICT_Y['K+']
    df = pd.read_csv(csv_data_path, encoding='utf-8-sig')

    total_data = np.array(df[VAR_LIST[1:-1]])

    labeled_index = []
    for i in range(total_data.shape[0]):
        if total_data[i][y_index] != -1:
            labeled_index.append(i)
    labeled_data = total_data[labeled_index, :]
    print(labeled_data.shape)

    split_by_bin(labeled_data[:, :var_num], labeled_data[:, y_index],
                 train_ratio=train_ratio,
                 bin_width=bin_width, inside_shuffle=False, plot=True)


# if __name__ == '__main__':
#     csv_data_path = r'../A2_data/202201/512/1月512有标签加无标签_v3.csv'
#     data_distribution(csv_data_path, 0.6, 1)
