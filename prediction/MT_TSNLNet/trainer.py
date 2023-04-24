#!coding:utf-8
import csv
import random

import torch
from torch.nn import functional as F
from sklearn.preprocessing import MinMaxScaler
import os
from datetime import datetime
from pathlib import Path
from collections import defaultdict

from prediction.MT_TSNLNet.utils.loss import SNP_TNPLoss, SNP_TNPLoss_v2
from prediction.MT_TSNLNet.utils.ramps import exp_rampup
from prediction.MT_TSNLNet.utils.datasets import decode_label
from prediction.MT_TSNLNet.utils.data_utils import NO_LABEL
from prediction.utils.metrics import cal_metrics, cal_metrics_avg
from prediction.utils.visualization import draw_pred_curve, Plot
import numpy as np
import pandas as pd


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
    return np.array(ED)


class Trainer:
    def __init__(self, model, ema_model, optimizer, device, scaler_y, config, train_X):
        if config.seed is not None:
            print('固定随机种子：{}'.format(config.seed))
            torch.manual_seed(config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(config.seed)
                torch.cuda.manual_seed_all(config.seed)
            np.random.seed(config.seed)

        self.model = model
        self.ema_model = ema_model
        self.optimizer = optimizer
        self.sup_loss = torch.nn.MSELoss(reduction='sum')
        self.cons_loss = torch.nn.MSELoss(reduction='sum')
        self.recon_loss = SNP_TNPLoss

        self.save_dir = '{}-{}-{}_{}'.format(config.arch, config.model, config.num_labels,
                                             config.time)
        self.save_dir = os.path.join(config.save_dir, self.save_dir)

        self.cons_loss_weight = config.cons_loss_weight
        self.recon_loss_weight = config.recon_loss_weight

        self.ema_decay = config.ema_decay
        self.rampup = exp_rampup(config.weight_rampup)
        self.save_freq = config.save_freq
        self.print_freq = config.print_freq
        self.device = device
        self.global_step = 0
        self.epoch = 0

        self.config = config
        self.scaler_y = scaler_y

        self.total_X = train_X

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

        self.hidden_X = torch.zeros((self.total_X.shape[0], self.config.net_struct[-1])).to(torch.float32)

    def train(self, data_loader, print_freq=20):
        self.model.train()
        self.ema_model.train()
        with torch.enable_grad():
            return self.train_iteration(data_loader, print_freq)

    def train_iteration(self, data_loader, print_freq):
        loop_info = defaultdict(list)
        label_n, unlab_n = 0, 0
        for batch_idx, data in enumerate(data_loader):
            self.global_step += 1
            inputs = data[0][0].to(self.device)
            idxs = data[0][1].to(self.device)
            targets = data[1].to(self.device).reshape(-1, 1)
            ##=== decode targets ===
            lmask, umask = self.decode_targets(targets)
            lbs, ubs = lmask.float().sum().item(), umask.float().sum().item()

            ##=== forward ===
            outputs, hidden = self.model(inputs)
            sup_loss = self.sup_loss(outputs[lmask], targets[lmask])
            loop_info['lloss1'].append(sup_loss.item())

            # 计算K近邻的映射
            # with torch.no_grad():
            bs = outputs.size()[0]
            spa_knn_idx = self.spatial_knn_index[idxs, :].reshape(-1)
            tmp_knn_idx = self.temporal_knn_index[idxs, :].reshape(-1)
            knn_idxs = torch.cat((spa_knn_idx, tmp_knn_idx), dim=0).long()

            output = self.model(self.total_X[knn_idxs, :])[1]
            spa_knn_output = output[:bs * self.K_s, :].reshape(bs, self.K_s, -1)
            tem_knn_output = output[bs * self.K_s:, :].reshape(bs, self.K_t, -1)
            spa_w = self.spatial_M[idxs, :].reshape(-1)
            tem_w = self.temporal_M[idxs, :].reshape(-1)

            snp_loss = SNP_TNPLoss_v2(hidden, spa_knn_output, spa_w, mean=False)
            tnp_loss = SNP_TNPLoss_v2(hidden, tem_knn_output, tem_w, mean=False)

            # snp_loss = self.recon_loss(hidden, idxs, self.total_X, self.spatial_knn_index, self.spatial_M)
            # tnp_loss = self.recon_loss(hidden, idxs, self.total_X, self.temporal_knn_index, self.temporal_M)
            recon_loss = self.snp_loss_weight * snp_loss + self.tnp_loss_weight * tnp_loss
            recon_loss *= self.rampup(self.epoch) * self.recon_loss_weight
            # recon_loss *= self.recon_loss_weight
            loop_info['lloss2'].append(recon_loss.item())

            ##=== MT_TSNLNet Training ===
            self.update_ema(self.model, self.ema_model, self.ema_decay, self.global_step)

            ## consistency loss
            with torch.no_grad():
                ema_outputs = self.ema_model(inputs)[0]
                ema_outputs = ema_outputs.detach()

            cons_loss = self.cons_loss(outputs, ema_outputs)
            cons_loss *= self.rampup(self.epoch) * self.cons_loss_weight
            loop_info['aCons'].append(cons_loss.item())

            loss = sup_loss + cons_loss + recon_loss

            ## backwark
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            ##=== log info ===
            label_n, unlab_n = label_n + lbs, unlab_n + ubs
            # if print_freq > 0 and (batch_idx % print_freq) == 0:
            #     print(f"[train][{batch_idx:<3}]", self.gen_info(loop_info, lbs, ubs))
        # print(f">>>[train]", self.gen_info(loop_info, label_n, unlab_n, False))
        return loop_info, label_n

    def test(self, data_loader, plot=False):
        self.model.eval()
        self.ema_model.eval()
        with torch.no_grad():
            return self.test_iteration(data_loader, plot=plot)

    def test_iteration(self, data_loader, plot=False):
        loop_info = defaultdict(list)
        label_n, unlab_n = 0, 0
        for batch_idx, (data, targets) in enumerate(data_loader):
            targets = targets.reshape(-1, 1)
            data, targets = data.to(self.device), targets.to(self.device)
            lbs, ubs = data.size(0), -1

            ##=== forward ===
            outputs = self.model(data)[0]
            ema_outputs = self.ema_model(data)[0]
            loss = self.sup_loss(outputs, targets)

            loop_info['lloss'].append(loss.item())

            real_y = self.scaler_y.inverse_transform(targets.cpu().reshape(-1, 1))
            pred_y = self.scaler_y.inverse_transform(outputs.cpu().reshape(-1, 1))
            ema_pred_y = self.scaler_y.inverse_transform(ema_outputs.cpu().reshape(-1, 1))

            # real_y_v2 = []
            # pred_y_v2 = []
            # ema_pred_y_v2 = []
            # start = 0
            # for i in range(real_y.shape[0] - 1):
            #     if real_y[i][0] != real_y[i + 1][0] or i == real_y.shape[0] - 2:
            #         real_y_v2.append(np.mean(real_y[start: i + 1, :]))
            #         pred_y_v2.append(np.mean(pred_y[start: i + 1, :]))
            #         ema_pred_y_v2.append(np.mean(ema_pred_y[start: i + 1, :]))
            #         start = i + 1
            # real_y_v2 = np.array(real_y_v2).reshape(-1, 1)
            # pred_y_v2 = np.array(pred_y_v2).reshape(-1, 1)
            # ema_pred_y_v2 = np.array(ema_pred_y_v2).reshape(-1, 1)

            smodel_res = cal_metrics(real_y, pred_y, print_info=False)
            loop_info['mae_s'] = smodel_res['mae']
            loop_info['mape_s'] = smodel_res['mape']
            loop_info['mse_s'] = smodel_res['mse']
            loop_info['rmse_s'] = smodel_res['rmse']
            loop_info['r2_s'] = smodel_res['r2']
            loop_info['shot2_5_s'] = smodel_res['shot2_5']
            loop_info['shot5_s'] = smodel_res['shot5']
            loop_info['shot10_s'] = smodel_res['shot10']

            tmodel_res = cal_metrics(real_y, ema_pred_y, print_info=False)
            loop_info['mae_t'] = tmodel_res['mae']
            loop_info['mape_t'] = tmodel_res['mape']
            loop_info['mse_t'] = tmodel_res['mse']
            loop_info['rmse_t'] = tmodel_res['rmse']
            loop_info['r2_t'] = tmodel_res['r2']
            loop_info['shot2_5_t'] = tmodel_res['shot2_5']
            loop_info['shot5_t'] = tmodel_res['shot5']
            loop_info['shot10_t'] = tmodel_res['shot10']

            smodel_res_v2, real_y_v2, pred_y_v2 = cal_metrics_avg(real_y, pred_y, print_info=False)
            loop_info['mae_s_v2'] = smodel_res_v2['mae']
            loop_info['mape_s_v2'] = smodel_res_v2['mape']
            loop_info['mse_s_v2'] = smodel_res_v2['mse']
            loop_info['rmse_s_v2'] = smodel_res_v2['rmse']
            loop_info['r2_s_v2'] = smodel_res_v2['r2']
            loop_info['shot2_5_s_v2'] = smodel_res_v2['shot2_5']
            loop_info['shot5_s_v2'] = smodel_res_v2['shot5']
            loop_info['shot10_s_v2'] = smodel_res_v2['shot10']

            tmodel_res_v2, _, ema_pred_y_v2 = cal_metrics_avg(real_y, ema_pred_y, print_info=False)
            loop_info['mae_t_v2'] = tmodel_res_v2['mae']
            loop_info['mape_t_v2'] = tmodel_res_v2['mape']
            loop_info['mse_t_v2'] = tmodel_res_v2['mse']
            loop_info['rmse_t_v2'] = tmodel_res_v2['rmse']
            loop_info['r2_t_v2'] = tmodel_res_v2['r2']
            loop_info['shot2_5_t_v2'] = tmodel_res_v2['shot2_5']
            loop_info['shot5_t_v2'] = tmodel_res_v2['shot5']
            loop_info['shot10_t_v2'] = tmodel_res_v2['shot10']

            if plot:
                model_out_path = Path(self.save_dir)
                save_path = os.path.join(model_out_path, '测试集结果_学生模型e{}.png'.format(self.epoch))
                draw_pred_curve(real_y, pred_y, title='test_set epoch：{}'.format(self.epoch),
                                fig_size=(14, 10), save_path=save_path)
                save_path = os.path.join(model_out_path, '测试集结果_教师模型e{}.png'.format(self.epoch))
                draw_pred_curve(real_y, ema_pred_y, title='test_set epoch：{}'.format(self.epoch),
                                fig_size=(14, 10), save_path=save_path)

                save_path = os.path.join(model_out_path, '测试集结果_学生模型取平均e{}.png'.format(self.epoch))
                Plot(real_y_v2, pred_y_v2, title='test_set epoch：{}'.format(self.epoch),
                     save_path=save_path)
                save_path = os.path.join(model_out_path, '测试集结果_教师模型取平均e{}.png'.format(self.epoch))
                Plot(real_y_v2, ema_pred_y_v2, title='test_set epoch：{}'.format(self.epoch),
                     save_path=save_path)

            label_n, unlab_n = label_n + lbs, unlab_n + ubs

        return loop_info, label_n

    def loop(self, epochs, train_data, test_data, scheduler=None):
        self.get_knn_mat()
        best_rmse, best_r2, best_epoch = 999, -999, 0
        model_type = None
        for ep in range(epochs):
            self.epoch = ep
            if scheduler is not None:
                scheduler.step()
            # print("------ Training epochs: {} ------".format(ep))
            self.train(train_data, self.print_freq)
            # print("------ Testing epochs: {} ------".format(ep))
            if (ep + 1) % (2 * self.save_freq) == 0 or ep + 1 == epochs:
                info, n = self.test(test_data, plot=True)
            else:
                info, n = self.test(test_data)

            rmse_s, r2_s = info['rmse_s_v2'], info['r2_s_v2']
            if rmse_s < best_rmse and r2_s > best_r2:
                best_rmse = rmse_s
                best_r2 = r2_s
                best_epoch = ep
                model_type = 'student model'
            rmse_t, r2_t = info['rmse_t_v2'], info['r2_t_v2']
            if rmse_t < best_rmse and r2_t > best_r2:
                best_rmse = rmse_t
                best_r2 = r2_t
                best_epoch = ep
                model_type = 'teacher model'

            # save model
            if self.save_freq != 0 and (ep + 1) % self.save_freq == 0:
                self.save(ep)  # 保存模型
                self.save_res(ep, info)  # 保存结果
                self.save_res_v2(ep, info)  # 保存结果
        print(">>>[best]   epoch:{}, {},  r2:{},  rmse:{}".format(best_epoch, model_type, best_r2, best_rmse))
        return best_rmse, best_r2

    def update_ema(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data * (1 - alpha))

    def decode_targets(self, targets):
        label_mask = targets.ge(0)
        unlab_mask = targets.le(NO_LABEL)
        targets[unlab_mask] = decode_label(targets[unlab_mask])
        return label_mask, unlab_mask

    def gen_info(self, info, lbs, ubs, iteration=True):
        ret = []
        nums = {'l': lbs, 'u': ubs, 'a': lbs + ubs}
        for k, val in info.items():
            n = nums[k[0]]
            v = val[-1] if iteration else sum(val)
            s = f'{k}: {v / n:.3%}' if k[-1] == 'c' else f'{k}: {v:.5f}'
            ret.append(s)
        return '\t'.join(ret)

    def save(self, epoch, **kwargs):
        if self.save_dir is not None:
            model_out_path = Path(self.save_dir)
            state = {"epoch": epoch,
                     "weight_s": self.model.state_dict(),
                     "weight_t": self.ema_model.state_dict()}
            if not model_out_path.exists():
                model_out_path.mkdir()
                self.save_log(config=self.config, path=os.path.join(model_out_path, 'super_para.txt'))
            save_target = model_out_path / "model_epoch_{}.pth".format(epoch)
            torch.save(state, save_target)
            print('==> save model to {}'.format(save_target))

    def save_log(self, config, path):
        with open(path, 'w', encoding='utf-8-sig') as f:
            for arg in vars(config):
                log_info = format(arg, '<20') + format(str(getattr(config, arg)), '<') + '\n'
                f.write(log_info)

    def save_res(self, epoch, res_info):
        model_out_path = Path(self.save_dir)
        file_path = os.path.join(model_out_path, 'super_para.txt')
        with open(file_path, encoding="utf-8-sig", mode="a") as file:
            file.write('\n\n--------epoch：{}--------'.format(epoch))
            file.write('\n学生模型：')
            file.write('\nmae：{}'.format(res_info['mae_s']))
            file.write('\nmape：{}'.format(res_info['mape_s']))
            file.write('\nmse：{}'.format(res_info['mse_s']))
            file.write('\nrmse：{}'.format(res_info['rmse_s']))
            file.write('\nr2：{}'.format(res_info['r2_s']))
            file.write('\n命中率2.5%：{}'.format(res_info['shot2_5_s']))
            file.write('\n命中率5%：{}'.format(res_info['shot5_s']))
            file.write('\n命中率10%：{}'.format(res_info['shot10_s']))

            s_res_list = [res_info['mae_s'], res_info['mape_s'], res_info['mse_s'], res_info['rmse_s'],
                          res_info['r2_s'], res_info['shot2_5_s'], res_info['shot5_s'], res_info['shot10_s'],
                          self.config.time, self.config.sup_batch_size, self.config.usp_batch_size, self.config.lr,
                          self.config.net_struct, self.config.dropout, '学生模型',
                          self.config.num_labels, self.config.model, self.epoch,
                          self.config.cons_loss_weight, self.config.recon_loss_weight, self.config.ema_decay,
                          self.config.lr_scheduler, self.config.min_lr,
                          self.config.K_t, self.config.delta_t, self.config.K_s, self.config.delta_s,
                          self.config.snp_loss_weight]

            file.write('\n\n教师模型：')
            file.write('\nmae：{}'.format(res_info['mae_t']))
            file.write('\nmape：{}'.format(res_info['mape_t']))
            file.write('\nmse：{}'.format(res_info['mse_t']))
            file.write('\nrmse：{}'.format(res_info['rmse_t']))
            file.write('\nr2：{}'.format(res_info['r2_t']))
            file.write('\n命中率2.5%：{}'.format(res_info['shot2_5_t']))
            file.write('\n命中率5%：{}'.format(res_info['shot5_t']))
            file.write('\n命中率10%：{}'.format(res_info['shot10_t']))

            t_res_list = [res_info['mae_t'], res_info['mape_t'], res_info['mse_t'], res_info['rmse_t'],
                          res_info['r2_t'], res_info['shot2_5_t'], res_info['shot5_t'], res_info['shot10_t'],
                          self.config.time, self.config.sup_batch_size, self.config.usp_batch_size, self.config.lr,
                          self.config.net_struct, self.config.dropout, '教师模型',
                          self.config.num_labels, self.config.model, self.epoch,
                          self.config.cons_loss_weight, self.config.recon_loss_weight, self.config.ema_decay,
                          self.config.lr_scheduler, self.config.min_lr,
                          self.config.K_t, self.config.delta_t, self.config.K_s, self.config.delta_s,
                          self.config.snp_loss_weight]

        with open(self.config.res_csv_path, mode='a', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(s_res_list)
            writer.writerow(t_res_list)

    def save_res_v2(self, epoch, res_info):
        model_out_path = Path(self.save_dir)
        file_path = os.path.join(model_out_path, '取平均结果.txt')
        with open(file_path, encoding="utf-8-sig", mode="a") as file:
            file.write('\n\n--------epoch：{}--------'.format(epoch))
            file.write('\n--------取平均 | 学生模型--------'.format(epoch))
            file.write('\nmae：{}'.format(res_info['mae_s_v2']))
            file.write('\nmape：{}'.format(res_info['mape_s_v2']))
            file.write('\nmse：{}'.format(res_info['mse_s_v2']))
            file.write('\nrmse：{}'.format(res_info['rmse_s_v2']))
            file.write('\nr2：{}'.format(res_info['r2_s_v2']))
            file.write('\n命中率2.5%：{}'.format(res_info['shot2_5_s_v2']))
            file.write('\n命中率5%：{}'.format(res_info['shot5_s_v2']))
            file.write('\n命中率10%：{}'.format(res_info['shot10_s_v2']))

            s_res_list = [res_info['mae_s_v2'], res_info['mape_s_v2'], res_info['mse_s_v2'], res_info['rmse_s_v2'],
                          res_info['r2_s_v2'], res_info['shot2_5_s_v2'], res_info['shot5_s_v2'],
                          res_info['shot10_s_v2'],
                          self.config.time, self.config.sup_batch_size, self.config.usp_batch_size, self.config.lr,
                          self.config.net_struct, self.config.dropout, '学生模型取平均',
                          self.config.num_labels, self.config.model, self.epoch,
                          self.config.cons_loss_weight, self.config.recon_loss_weight, self.config.ema_decay,
                          self.config.lr_scheduler, self.config.min_lr,
                          self.config.K_t, self.config.delta_t, self.config.K_s, self.config.delta_s,
                          self.config.snp_loss_weight]

            file.write('\n\n--------取平均 | 教师模型--------'.format(epoch))
            file.write('\nmae：{}'.format(res_info['mae_t_v2']))
            file.write('\nmape：{}'.format(res_info['mape_t_v2']))
            file.write('\nmse：{}'.format(res_info['mse_t_v2']))
            file.write('\nrmse：{}'.format(res_info['rmse_t_v2']))
            file.write('\nr2：{}'.format(res_info['r2_t_v2']))
            file.write('\n命中率2.5%：{}'.format(res_info['shot2_5_t_v2']))
            file.write('\n命中率5%：{}'.format(res_info['shot5_t_v2']))
            file.write('\n命中率10%：{}'.format(res_info['shot10_t_v2']))

            t_res_list = [res_info['mae_t_v2'], res_info['mape_t_v2'], res_info['mse_t_v2'], res_info['rmse_t_v2'],
                          res_info['r2_t_v2'], res_info['shot2_5_t_v2'], res_info['shot5_t_v2'],
                          res_info['shot10_t_v2'],
                          self.config.time, self.config.sup_batch_size, self.config.usp_batch_size, self.config.lr,
                          self.config.net_struct, self.config.dropout, '教师模型取平均',
                          self.config.num_labels, self.config.model, self.epoch,
                          self.config.cons_loss_weight, self.config.recon_loss_weight, self.config.ema_decay,
                          self.config.lr_scheduler, self.config.min_lr,
                          self.config.K_t, self.config.delta_t, self.config.K_s, self.config.delta_s,
                          self.config.snp_loss_weight]

        with open(self.config.res_csv_path, mode='a', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(s_res_list)
            writer.writerow(t_res_list)

    def get_knn_mat(self):
        time_series = self.total_X[:, 0]
        self.total_X = self.total_X[:, 1:].astype(float)

        # ----------------------------------------------------------------------------------------------
        if self.config.fusion:
            if self.config.img_size == 512:
                tem_dir_path = os.path.join(r'近邻矩阵/512/fusion_temporal_mat',
                                            '{}_{}_{}'.format(self.total_X.shape[0], self.K_t, self.delta_t))
            else:
                tem_dir_path = os.path.join(r'近邻矩阵/1388/fusion_temporal_mat',
                                            '{}_{}_{}'.format(self.total_X.shape[0], self.K_t, self.delta_t))
        else:
            if self.config.img_size == 512:
                tem_dir_path = os.path.join(r'近邻矩阵/512/temporal_mat',
                                            '{}_{}_{}'.format(self.total_X.shape[0], self.K_t, self.delta_t))
            else:
                tem_dir_path = os.path.join(r'近邻矩阵/1388/temporal_mat',
                                            '{}_{}_{}'.format(self.total_X.shape[0], self.K_t, self.delta_t))

        if os.path.exists(tem_dir_path):
            print('载入储存的时间近邻权重矩阵...{}'.format(tem_dir_path))
            df1 = pd.read_csv(os.path.join(tem_dir_path, 'temporal_knn_index.csv'))
            self.temporal_knn_index = np.array(df1)[:, 1:]
            df2 = pd.read_csv(os.path.join(tem_dir_path, 'temporal_M.csv'))
            self.temporal_M = np.array(df2)[:, 1:]

        else:
            ## 计算时间邻接图和权重矩阵
            print('计算时间近邻权重矩阵...')
            for j in range(time_series.shape[0]):
                time_series[j] = datetime.timestamp(time_series[j])
                time_series[j] -= 1641542340.
            time_series = time_series / 86400
            time_series = time_series.reshape(-1, 1)

            scaler_t = MinMaxScaler()
            time_series = scaler_t.fit_transform(time_series)

            # 计算时间邻接图和权重矩阵
            self.temporal_M = np.zeros((self.total_X.shape[0], self.K_t))
            self.temporal_knn_index = np.zeros((self.total_X.shape[0], self.K_t))
            for j in range(self.total_X.shape[0]):
                # print(j, end=' ')
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
        # ----------------------------------------------------------------------------------------------
        if self.config.fusion:
            if self.config.img_size == 512:
                spa_dir_path = os.path.join('近邻矩阵/512/fusion_spatial_mat',
                                            '{}_{}_{}_{}'.format(self.total_X.shape[0], self.K_s, self.delta_s,
                                                                 self.config.smooth_data))
            else:
                spa_dir_path = os.path.join('近邻矩阵/1388/fusion_spatial_mat',
                                            '{}_{}_{}_{}'.format(self.total_X.shape[0], self.K_s, self.delta_s,
                                                                 self.config.smooth_data))
        else:
            if self.config.img_size == 512:
                spa_dir_path = os.path.join('近邻矩阵/512/spatial_mat',
                                            '{}_{}_{}_{}'.format(self.total_X.shape[0], self.K_s, self.delta_s,
                                                                 self.config.smooth_data))
            else:
                spa_dir_path = os.path.join('近邻矩阵/1388/spatial_mat',
                                            '{}_{}_{}_{}'.format(self.total_X.shape[0], self.K_s, self.delta_s,
                                                                 self.config.smooth_data))

        if os.path.exists(spa_dir_path):
            print('载入储存的空间近邻权重矩阵...{}'.format(spa_dir_path))
            df1 = pd.read_csv(os.path.join(spa_dir_path, 'spatial_knn_index.csv'))
            self.spatial_knn_index = np.array(df1)[:, 1:]
            df2 = pd.read_csv(os.path.join(spa_dir_path, 'spatial_M.csv'))
            self.spatial_M = np.array(df2)[:, 1:]

        else:
            ## 计算空间邻接图和权重矩阵
            print('计算空间近邻权重矩阵...')
            self.spatial_knn_index = np.zeros((self.total_X.shape[0], self.K_s))
            self.spatial_M = np.zeros((self.total_X.shape[0], self.K_s))
            dis_mat = None
            batch_idx = 0
            bs = 1000
            for j in range(self.total_X.shape[0]):
                if j % bs == 0:
                    # print(j)
                    dis_mat = EuclideanDistances(self.total_X[batch_idx * bs:(batch_idx + 1) * bs, :], self.total_X)
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
        self.total_X = torch.from_numpy(self.total_X).to(torch.float32).to(self.device)
        self.temporal_knn_index = torch.from_numpy(self.temporal_knn_index).int().to(self.device)
        self.temporal_M = torch.from_numpy(self.temporal_M).to(torch.float32).to(self.device)
        self.spatial_knn_index = torch.from_numpy(self.spatial_knn_index).int().to(self.device)
        self.spatial_M = torch.from_numpy(self.spatial_M).to(torch.float32).to(self.device)
