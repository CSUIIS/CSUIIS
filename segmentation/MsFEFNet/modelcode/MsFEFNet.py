from torch import nn
import torch
import torch.nn.functional as F
from torchstat import stat


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, stride=1):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if self.in_channels != self.out_channels:
            self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                   bias=False)
            self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # 判断stride==2
        x_identity = x
        if self.in_channels != self.out_channels:
            x_identity = self.bn3(self.conv3(x))
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + x_identity
        x = self.relu(x)
        return x


def make_block(in_channels, block_num):
    layers = []
    for i in range(block_num):
        layers.append(DoubleConv(in_channels, in_channels))
    return nn.Sequential(*layers)


class Down(nn.Module):
    def __init__(self, in_channels, down_factor=2):
        super(Down, self).__init__()
        assert down_factor == 2 or down_factor == 4
        self.down_factor = down_factor
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels * 2, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(in_channels * 2)
        self.relu = nn.ReLU(inplace=True)
        if down_factor == 4:
            self.conv2 = nn.Conv2d(in_channels=in_channels * 2, out_channels=in_channels * 4, kernel_size=1)
            self.bn2 = nn.BatchNorm2d(in_channels * 4)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 改成平均池化

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        if self.down_factor == 4:
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.pool(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, up_factor=2, channel_red=True):
        super(Up, self).__init__()
        assert up_factor == 2 or up_factor == 4 or up_factor == 8
        self.up_factor = up_factor
        if channel_red:
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1, stride=1,
                                   bias=False)
            self.bn1 = nn.BatchNorm2d(in_channels // 2)
        else:
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1,
                                   stride=1,
                                   bias=False)
            self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        if self.up_factor == 4 or self.up_factor == 8:
            self.conv2 = nn.Conv2d(in_channels=in_channels // 2, out_channels=in_channels // 4, kernel_size=1, stride=1,
                                   bias=False)
            self.bn2 = nn.BatchNorm2d(in_channels // 4)
            self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        if self.up_factor == 8:
            self.conv3 = nn.Conv2d(in_channels=in_channels // 4, out_channels=in_channels // 8, kernel_size=1, stride=1,
                                   bias=False)
            self.bn3 = nn.BatchNorm2d(in_channels // 8)
            self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.up1(x)
        if self.up_factor == 4 or self.up_factor == 8:
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.up2(x)
        if self.up_factor == 8:
            x = self.relu(self.bn3(self.conv3(x)))
            x = self.up3(x)
        return x


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=2, dilation=2, bias=False)
        self.bn1 = nn.BatchNorm2d(1)
        # self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_pool, max_pool], dim=1)
        # relu改成sigmoid
        # x = self.relu(self.bn1(self.conv1(x)))
        x = self.sigmoid(self.conv1(x))
        return x


class Fusion(nn.Module):
    def __init__(self, scale_num):
        super(Fusion, self).__init__()
        self.scale_num = scale_num
        self.sa = SpatialAttention()
        self.fusion = nn.Conv2d(in_channels=scale_num, out_channels=scale_num, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(scale_num)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_list, plot=False):
        x = []
        for i in range(self.scale_num):
            x.append(self.sa(x_list[i]))

        x = torch.cat(x, dim=1)
        x = self.relu(self.bn(self.fusion(x)))

        # --------------------------
        if plot:
            draw_features(self.scale_num, 1, x.cpu().numpy(), "f1_conv1.png")
        # --------------------------

        # # 加softmax?
        out = x_list[0] * torch.unsqueeze(x[:, 0, :, :], dim=1)
        for i in range(1, self.scale_num):
            out = out + x_list[i] * torch.unsqueeze(x[:, i, :, :], dim=1)
        return out


block_nums = [3, 3, 2]


# def draw_features(width, height, x, savename):
#     import matplotlib.pyplot as plt
#     import numpy as np
#     import cv2
#     fig = plt.figure(figsize=(16, 16))
#     fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
#     for i in range(width * height):
#         plt.subplot(height, width, i + 1)
#         plt.axis('off')
#         img = x[0, i, :, :]
#         # img = x[0,i, :, :]
#         pmin = np.min(img)
#         pmax = np.max(img)
#         print(pmin)
#         print(pmax)
#         img = ((img - pmin) / (pmax - pmin)) * 255  # float在[0，1]之间，转换成0-255
#         img = img.astype(np.uint8)  # 转成unit8
#         img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # 生成heat map
#         img = img[:, :, ::-1]  # 注意cv2（BGR）和matplotlib(RGB)通道是相反的
#         plt.imshow(img)
#         print("{}/{}".format(i, width * height))
#     # fig.savefig(savename,bbox_inches = 'tight')
#     # fig.clf()
#     plt.show()


class MsFEFNet(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 2,
                 base_c: int = 32):
        super(MsFEFNet, self).__init__()
        self.conv1 = DoubleConv(in_channels=in_channels, out_channels=64)
        self.down = DoubleConv(in_channels=64, out_channels=128, stride=2)

        self.conv2 = nn.Conv2d(in_channels=128, out_channels=base_c, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=2 * base_c, kernel_size=3, padding=1, stride=2, bias=False)
        self.bn2 = nn.BatchNorm2d(2 * base_c)
        self.block2_1 = make_block(1 * base_c, block_nums[0])
        self.block2_2 = make_block(2 * base_c, block_nums[0])
        self.down_s2_1t2 = Down(in_channels=base_c, down_factor=2)
        self.down_s2_2t3 = Down(in_channels=base_c * 2, down_factor=2)
        self.up_s2_2t1 = Up(in_channels=base_c * 2, up_factor=2)

        self.fusion_s2_1 = Fusion(2)
        self.fusion_s2_2 = Fusion(2)

        self.block3_1 = make_block(1 * base_c, block_nums[1])
        self.block3_2 = make_block(2 * base_c, block_nums[1])
        self.block3_3 = make_block(4 * base_c, block_nums[1])
        self.down_s3_1t2 = Down(in_channels=base_c, down_factor=2)
        self.down_s3_1t3 = Down(in_channels=base_c, down_factor=4)
        self.down_s3_2t3 = Down(in_channels=base_c * 2, down_factor=2)
        self.down_s3_3t4 = Down(in_channels=base_c * 4, down_factor=2)
        self.up_s3_2t1 = Up(in_channels=base_c * 2, up_factor=2)
        self.up_s3_3t1 = Up(in_channels=base_c * 4, up_factor=4)
        self.up_s3_3t2 = Up(in_channels=base_c * 4, up_factor=2)

        self.fusion_s3_1 = Fusion(3)
        self.fusion_s3_2 = Fusion(3)
        self.fusion_s3_3 = Fusion(3)

        self.block4_1 = make_block(1 * base_c, block_nums[2])
        self.block4_2 = make_block(2 * base_c, block_nums[2])
        self.block4_3 = make_block(4 * base_c, block_nums[2])
        self.block4_4 = make_block(8 * base_c, block_nums[2])
        self.up_s4_2t1 = Up(in_channels=base_c * 2, up_factor=2)
        self.up_s4_3t1 = Up(in_channels=base_c * 4, up_factor=4)
        self.up_s4_4t1 = Up(in_channels=base_c * 8, up_factor=8)

        self.fusion_s4_1 = Fusion(4)

        self.up = Up(in_channels=base_c, up_factor=2, channel_red=False)
        self.conv4 = DoubleConv(in_channels=base_c, out_channels=base_c)
        self.conv5 = nn.Conv2d(in_channels=base_c, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)

        x = self.down(x)
        x1 = self.relu(self.bn1(self.conv2(x)))
        x_skip = x1
        x2 = self.relu(self.bn2(self.conv3(x)))
        x1 = self.block2_1(x1)
        x1 = x1 + x_skip
        x2 = self.block2_2(x2)

        x1_copy = x1
        x2_copy = x2
        x1 = self.fusion_s2_1([x1_copy, self.up_s2_2t1(x2_copy)])

        x_skip = x1
        x2 = self.fusion_s2_2([self.down_s2_1t2(x1_copy), x2_copy])
        x3 = self.down_s2_2t3(x2_copy)
        x1 = self.block3_1(x1)
        x1 = x1 + x_skip
        x2 = self.block3_2(x2)
        x3 = self.block3_3(x3)

        x1_copy = x1
        x2_copy = x2
        x3_copy = x3
        x1 = self.fusion_s3_1([x1_copy, self.up_s3_2t1(x2_copy), self.up_s3_3t1(x3_copy)])
        x_skip = x1
        x2 = self.fusion_s3_2([self.down_s3_1t2(x1_copy), x2_copy, self.up_s3_3t2(x3_copy)])
        x3 = self.fusion_s3_3([self.down_s3_1t3(x1_copy), self.down_s3_2t3(x2_copy), x3_copy])
        x4 = self.down_s3_3t4(x3_copy)

        x1 = self.block4_1(x1)
        x1 = x1 + x_skip
        x2 = self.block4_2(x2)
        x3 = self.block4_3(x3)
        x4 = self.block4_4(x4)

        import time
        a = time.clock()
        for i in range(10):
            x1 = self.fusion_s4_1([x1, self.up_s4_2t1(x2), self.up_s4_3t1(x3), self.up_s4_4t1(x4)])
        b = time.clock()
        print(b - a)
        x1 = self.up(x1)
        x1 = self.conv4(x1)
        x1 = self.conv5(x1)
        return x1

    def count(self):
        print('MsFE 1(a)', count_param(self.block2_1))
        print('MsFE 1(b)', count_param(self.block2_2))
        print('MsFF 1-1', count_param(self.fusion_s2_1) + count_param(self.up_s2_2t1))
        print('MsFF 1-2', count_param(self.fusion_s2_2) + count_param(self.down_s2_1t2))

        print('MsFE 2(a)', count_param(self.block3_1))
        print('MsFE 2(b)', count_param(self.block3_2))
        print('MsFE 2(c)', count_param(self.block3_3))

        print('MsFF 2-1', count_param(self.fusion_s3_1) +
              count_param(self.up_s3_2t1) +
              count_param(self.up_s3_3t1))
        print('MsFF 2-2', count_param(self.fusion_s3_1) +
              count_param(self.down_s3_1t2) +
              count_param(self.up_s3_3t2))
        print('MsFF 2-3', count_param(self.fusion_s3_1) +
              count_param(self.down_s3_1t3) +
              count_param(self.down_s3_2t3))

        print('MsFE 3(a)', count_param(self.block4_1))
        print('MsFE 3(b)', count_param(self.block4_2))
        print('MsFE 3(c)', count_param(self.block4_3))
        print('MsFE 3(d)', count_param(self.block4_4))
        print('MsFF 3', count_param(self.fusion_s4_1) +
              count_param(self.up_s4_2t1) +
              count_param(self.up_s4_3t1) +
              count_param(self.up_s4_4t1))


def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count


if __name__ == '__main__':
    model = MsFEFNet(base_c=64)
    model.count()
    # print(count_param(model))
    # stat(model,(3,512,512))
