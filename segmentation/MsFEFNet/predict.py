'''
1、修改结果存放文件夹
2、修改是否保存文件：save_flag
3、修改模型路径，注意模型初始化参数是否一致:例如base_c
'''


import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from segmentation.utils.dataset import MyDatasetMeo
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import csv
from train import get_transform
import cv2 as cv
from segmentation.MsFEFNet.modelcode.MsFEFNet import MsFEFNet
import segmentation.utils.distributed_utils as utils


def open_recon(img_bin,kernel_size):
    kernal = np.ones((kernel_size, kernel_size), np.uint8)
    img_erode = cv.erode(img_bin, kernal)
    marker = img_erode.copy()
    mask = img_bin.copy()
    while True:
        marker_pre = marker
        dilation = cv.dilate(marker, kernal)
        marker = np.min((dilation, mask), axis=0)
        if (marker_pre == marker).all():
            break
    return marker


def print_info(bubble_size):
    num = len(bubble_size)
    print('---------------------------------------------')
    print('泡沫个数：{}'.format(num))
    print('最大尺寸：{}'.format(np.max(bubble_size)))
    print('最小尺寸：{}'.format(np.min(bubble_size)))
    print('平均尺寸：{}'.format(np.average(bubble_size)))
    print('标准差  ：{}'.format(np.std(bubble_size)))
    print('---------------------------------------------')


def single(img):
    kernel = np.ones((3, 3), np.uint8)
    img_recon = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    img_recon = open_recon(img, 9)
    img_recon = open_recon(img_recon, 3)
    retval, markers, stats, centroids = cv.connectedComponentsWithStats(img_recon, connectivity=4)
    bubble_size = []
    cut_num = 0
    for i in range(1, stats.shape[0]):
        lu_x, lu_y = stats[i][0], stats[i][1]
        w, h = stats[i][2], stats[i][3]
        area = stats[i][4]
        center_x, center_y = centroids[i][0], centroids[i][1]
        if area > 20:
            bubble_size.append(area)

    result = [0] * (max(bubble_size) - 21 + 1)
    for i in range(21, max(bubble_size) + 1):
        result[i - 21] = bubble_size.count(i)
    # plt.plot(A3_result)
    # plt.show()

    bubble_size = np.array(bubble_size)
    print_info(bubble_size)

    colors = []
    for i in range(retval):
        b = np.random.randint(0, 256)
        g = np.random.randint(0, 256)
        r = np.random.randint(0, 256)
        colors.append((b, g, r))
    colors[0] = (0, 0, 0)
    h, w = img_recon.shape
    image_color = np.zeros((h, w, 3), dtype=np.uint8)
    for row in range(h):
        for col in range(w):
            image_color[row, col] = colors[markers[row, col]]
    # plt.subplot(2,2,1)
    # plt.imshow(img)
    # plt.subplot(2,2,2)
    # plt.imshow(img_recon,cmap='gray')
    # plt.subplot(2,2,3)
    # plt.imshow(markers)
    # plt.subplot(2,2,4)
    # plt.imshow(image_color)
    # plt.show()
    return list(bubble_size)


if __name__ == '__main__':
    resdir_path = '../model_result/dataset1_valset/testnet_noMSFF_model18'  # 存放模型预测结果
    save_flag = False
    # 最优模型
    model_weight_path = 'model/model5_16/testnet_9_best.pkl'
    net = MsFEFNet(base_c=64)

    confmat = utils.ConfusionMatrix(2)
    actual_size_total = []
    predict_size_total = []

    imgdir_path = '../data/dataset1/test/img'
    gtdir_path = '../data/dataset1/test/label'

    if not os.path.exists(resdir_path):
        os.makedirs(resdir_path)
    imgs = []
    img_name_list = os.listdir(imgdir_path)
    for img_name in img_name_list:
        label_name = img_name.replace('.png', '_label.png')
        img_path = os.path.join(imgdir_path, img_name)
        gt_path = os.path.join(gtdir_path, label_name)

        img = Image.open(img_path).convert('RGB')
        gt = Image.open(gt_path)  # 1 二值图像 L 灰度图像
        imgs.append((img, gt))

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'

    test_set = MyDatasetMeo(imgs, transform=get_transform(train=False, mean=mean, std=std))
    test_num = len(test_set)

    batch_size = 1
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    print("using {} images for testing".format(test_num))

    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    net.to(device)

    # net = torch.load(r'model1_t1/testnet_6_1.pkl',map_location='cpu')

    net.eval()
    index = 0
    with torch.no_grad():
        for test_img, test_label in test_loader:
            import time
            a = time.clock()
            for i in range(10):
                output = net(test_img.to(device))
            b = time.clock()
            print(b - a)
            exit(0)

            gt = torch.squeeze(test_label)

            output = net(test_img.to(device))

            # vis = np.array(torch.squeeze(vis))
            # print(vis[1].shape)

            # test_img = torch.squeeze(test_img)
            # test_img = test_img / 2 + 0.5
            # npimg = test_img.numpy()
            # plt.subplot(1,2,1)
            # plt.imshow(np.transpose(npimg, (1, 2, 0)))
            # plt.subplot(1,2,2)
            # plt.imshow(img)
            # plt.show()

            pre = output
            pre = torch.squeeze(pre)
            pre = torch.argmax(pre, dim=0)

            gt_torch = gt.flatten()
            pre_torch = pre.flatten()
            confmat.update(pre_torch, gt_torch)

            gt = gt.cpu().numpy().astype(np.uint8)
            pre = pre.cpu().numpy().astype(np.uint8)

            actual_size = single(gt)
            predict_size = single(pre)
            actual_size_total.extend(actual_size)
            predict_size_total.extend(predict_size)

            test_img = torch.squeeze(test_img)
            test_img = test_img / 2 + 0.5
            npimg = test_img.numpy()

            # plt.subplot(1, 3, 1)
            # plt.title('{}_{}'.format(index+1,img_name_list[index]))
            # plt.imshow(np.transpose(npimg, (1, 2, 0)))
            # plt.subplot(1, 3, 2)
            # plt.title('gt_{}'.format(img_name_list[index]))
            # plt.imshow(gt, cmap='gray')
            # plt.subplot(1,3,3)
            # plt.title('pre_{}'.format(img_name_list[index]))
            # plt.imshow(A1_pre,cmap='gray')
            # plt.show()

            if save_flag:
                # cv.imwrite(os.path.join(resdir_path,img_name_list[index].replace('.png','_predict.png')),A1_pre)
                pre[pre!=0]=255
                cv.imwrite(os.path.join(resdir_path, img_name_list[index].replace('.png', '_show.png')), pre)
            index += 1
            print('------[{}/{}]------'.format(index,test_num))



        print('实际气泡个数：{}, 预测气泡个数：{}'.format(len(actual_size_total), len(predict_size_total)))
        x = sum(actual_size_total) / len(actual_size_total)
        x0 = sum(predict_size_total) / len(predict_size_total)
        print('x-x0:{}'.format(x - x0))
        print('x:{}, x0:{}'.format(x, x0))
        e_r = (x - x0) / x
        print('e_r: ', e_r)
        print(confmat)

