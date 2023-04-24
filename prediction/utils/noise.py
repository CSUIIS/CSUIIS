import numpy as np
import random


def cosine_similarity(x, y):
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    # return 0.5 * (num / denom) + 0.5
    return num / denom


# 根据样本加噪
def cos_sim_noise(data_X, data_y, size_index=12, bias_factor=2):
    assert data_X.shape[0] == data_y.shape[0],'X和y长度不一致'

    y_list = []
    for i in range(data_y.shape[0]):
        if data_y[i] not in y_list:
            y_list.append(data_y[i])
    data_by_y = []
    for i in range(len(y_list)):
        data_by_y.append([])

    for i in range(data_X.shape[0]):
        index = y_list.index(data_y[i])
        data_by_y[index].append(list(data_X[i]))

    total_data = None
    for i in range(len(data_by_y)):
        # 获取每一个y对应的数据
        temp_data_pro = np.array(data_by_y[i])
        prototype = np.mean(temp_data_pro, axis=0).reshape(1, -1)

        y_list[i] = [y_list[i]] * temp_data_pro.shape[0]
        temp_y = np.array(y_list[i])
        temp_data_pro = np.append(temp_data_pro, temp_y, axis=1)

        for j in range(temp_data_pro.shape[0]):
            mean_avgsize = prototype[0][size_index]
            row_data = temp_data_pro[j][:-1]
            sim = cosine_similarity(row_data,prototype)[0]
            # sim = r2_score(row_data.reshape(-1),prototype.reshape(-1))
            bias = bias_factor * (1 - sim)
            sign = 1 if temp_data_pro[j][12] < mean_avgsize else -1
            bias = bias * sign
            temp_data_pro[j][-1] = temp_data_pro[j][-1] + bias
        if i == 0:
            total_data = temp_data_pro
        else:
            total_data = np.append(total_data, temp_data_pro, axis=0)
    new_data_X = total_data[:, :-1]
    new_data_y = total_data[:, -1].reshape(-1, 1)

    return new_data_X,new_data_y


def gauss_noise(data_y,mu=0,sigma=0.1,seed=None):
    data_y = data_y.reshape(-1)
    if seed is not None:
        random.seed(seed)
    for i in range(data_y.shape[0]):
        rand = random.gauss(mu,sigma)
        data_y[i] += rand
    return data_y.reshape(-1,1)


def random_noise(data_y, n_range=0.1,seed=None):
    data_y = data_y.reshape(-1)
    if seed is not None:
        np.random.seed(seed)
    for i in range(data_y.shape[0]):
        rand = np.random.uniform(-n_range,n_range)
        data_y[i] += rand
    return data_y.reshape(-1,1)