import scipy.signal
import matplotlib.pyplot as plt
# from pykalman import KalmanFilter
import numpy as np


def savgol_filter_A(data_X, window_size, poly_order=5, plot=False):
    for i in range(0, data_X.shape[1]):
        col_data = data_X[:, i]
        smoothed_data = scipy.signal.savgol_filter(col_data, window_size, poly_order)
        if plot:
            plt.plot(col_data)
            plt.plot(smoothed_data)
            plt.title('column:{}'.format(i + 1))
            plt.show()
        data_X[:, i] = smoothed_data.copy()
    return data_X


def karman_filter_A(data_X, damping, plot=False):
    for i in range(0, data_X.shape[1]):
        col_data = data_X[:, i]
        smoothed_data = kalman_1d(col_data, damping=damping).reshape(-1)
        if plot:
            plt.plot(col_data)
            plt.plot(smoothed_data)
            plt.title('column:{}'.format(i + 1))
            plt.show()
        data_X[:, i] = smoothed_data.copy()
    return data_X


def kalman_1d(observations, damping=1):
    # To return the smoothed time series data
    observation_covariance = damping
    initial_value_guess = observations[0]
    transition_matrix = 1
    transition_covariance = 0.1
    initial_value_guess
    kf = KalmanFilter(
        initial_state_mean=initial_value_guess,
        initial_state_covariance=observation_covariance,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance,
        transition_matrices=transition_matrix
    )
    pred_state, state_cov = kf.smooth(observations)
    return pred_state


# 单标签内平滑
def savgol_filter_B(data_X, data_y, window_size, poly_order=5):
    assert data_X.shape[0] == data_y.shape[0], 'X和y的长度不一致'

    y_list = []
    for i in range(data_X.shape[0]):
        if list(data_y[i, :]) not in y_list:
            y_list.append(list(data_y[i, :]))

    data_by_y = []
    for i in range(len(y_list)):
        data_by_y.append([])

    for i in range(data_X.shape[0]):
        index = y_list.index(list(data_y[i, :]))
        data_by_y[index].append(list(data_X[i, :]))

    total_data = None
    for i in range(len(data_by_y)):
        temp_data_pro = np.array(data_by_y[i])

        for j in range(0, temp_data_pro.shape[1]):
            col_data = temp_data_pro[:, j]
            smoothed_data = scipy.signal.savgol_filter(col_data, window_size, poly_order)
            # plt.plot(col_data)
            # plt.plot(smoothed_data)
            # plt.show()
            temp_data_pro[:, j] = smoothed_data.copy()

        temp_y = [y_list[i]] * temp_data_pro.shape[0]
        temp_y = np.array(temp_y)

        temp_data_pro = np.append(temp_data_pro, temp_y, axis=1)
        if i == 0:
            total_data = temp_data_pro
        else:
            total_data = np.append(total_data, temp_data_pro, axis=0)

    return total_data


def karman_filter_B(data_X, data_y, damping):
    assert data_X.shape[0] == data_y.shape[0], 'X和y的长度不一致'
    y_list = []
    for i in range(data_X.shape[0]):
        if list(data_y[i, :]) not in y_list:
            y_list.append(list(data_y[i, :]))

    data_by_y = []
    for i in range(len(y_list)):
        data_by_y.append([])

    for i in range(data_X.shape[0]):
        index = y_list.index(list(data_y[i, :]))
        data_by_y[index].append(list(data_X[i, :]))

    total_data = None
    for i in range(len(data_by_y)):
        temp_data_pro = np.array(data_by_y[i])

        for j in range(0, temp_data_pro.shape[1]):
            col_data = temp_data_pro[:, j]
            smoothed_data = kalman_1d(col_data, damping=damping).reshape(-1)
            # plt.plot(col_data)
            # plt.plot(smoothed_data)
            # plt.show()
            temp_data_pro[:, j] = smoothed_data.copy()

        temp_y = [y_list[i]] * temp_data_pro.shape[0]
        temp_y = np.array(temp_y)

        temp_data_pro = np.append(temp_data_pro, temp_y, axis=1)
        if i == 0:
            total_data = temp_data_pro
        else:
            total_data = np.append(total_data, temp_data_pro, axis=0)

    return total_data
