from PIL import Image
from torch.utils.data import Dataset,DataLoader
import os
import sys
import torch
import segmentation.utils.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models.resnet
from torch import nn,optim
from tqdm import tqdm
from segmentation.utils.dataset import MyDatasetMeo
import csv
import cv2 as cv
from segmentation.MsFEFNet.modelcode.MsFENet import MsFENet
from segmentation.utils.train_and_eval import train_one_epoch,evaluate,criterion


def imshow(img):
    img = torch.squeeze(img)
    img = img/2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()


def focal_loss(output,label,alpha=0.25,gamma=2.0,reduction='mean'):    # alpha=0.25
    alpha = torch.tensor(alpha).to(device)
    gamma = torch.tensor(gamma).to(device)
    batch_size,*_,height,width = output.size()
    loss = torch.zeros(batch_size).to(device)
    f_loss = torch.tensor(0.0).to(device)
    for i in range(batch_size):
        single_img = output[i].clone().view(1,-1)
        single_label = label[i].clone().view(1,-1)
        single_img[single_label!=1] = 1 - single_img[single_label!=1]   #pt
        loss[i] = torch.pow((1-single_img),gamma).mm(torch.log(single_img).t())
        loss[i] = -alpha * loss[i].clone()
    if reduction == 'mean':
        for i in range(len(loss)):
            loss[i] = loss[i]/(height*width)
        f_loss = sum(loss)/len(loss)
    if reduction == 'sum':
        f_loss = sum(loss)
    return f_loss


class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.7 * base_size)
        max_size = int(1.2 * base_size)

        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.CenterCrop(512),  # 自己加的
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    base_size = 512
    crop_size = 512

    if train:
        return SegmentationPresetTrain(base_size, crop_size, hflip_prob=0,vflip_prob=0,mean=mean, std=std)
    else:
        return SegmentationPresetEval(mean=mean, std=std)


def adjust_learning_rate(optimizer, epoch):
    global lr
    lr2 = lr * (0.8 ** (epoch // 30))
    print(lr2)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr2


if __name__ == '__main__':
    imgdir_path = '../data/dataset1/train/img'
    labeldir_path = '../data/dataset1/train/label'
    imgs = []
    img_name_list = os.listdir(imgdir_path)
    # img_name_list = img_name_list[0:20]
    for img_name in img_name_list:
        label_name = img_name.replace('.png','_label.png')
        img_path = os.path.join(imgdir_path,img_name)
        label_path = os.path.join(labeldir_path,label_name)

        img = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('L')   # 1 二值图像 L 灰度图像

        # plt.figure(0)
        # plt.subplot(1,2,1)
        # plt.imshow(img)
        # plt.subplot(1,2,2)
        # plt.imshow(label)
        # plt.show()
        imgs.append((img, label))

    val_imgdir_path = '../data/dataset1/validation/img'
    val_labeldir_path = '../data/dataset1/validation/label'
    val_imgs = []
    val_img_name_list = os.listdir(val_imgdir_path)
    # val_img_name_list = val_img_name_list[0:6]
    for img_name in val_img_name_list:
        label_name = img_name.replace('.png', '_label.png')
        img_path = os.path.join(val_imgdir_path, img_name)
        label_path = os.path.join(val_labeldir_path, label_name)
        img = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('L')  # 1 二值图像 L 灰度图像
        val_imgs.append((img, label))

    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    # mean=[0.7315411,0.75076623,0.72877709]123456
    # std=[0.03975445,0.03883997,0.04397608]
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    train_set = MyDatasetMeo(imgs,transform=get_transform(train=True,mean=mean,std=std))
    train_num = len(train_set)

    val_set = MyDatasetMeo(val_imgs, transform=get_transform(train=False, mean=mean, std=std))
    val_num = len(val_set)

    batch_size = 6
    lr = 0.001
    epochs = 400
    acc_steps = 1
    # momentum = 0.9

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,pin_memory=True)

    # for img, label_fromLabelme in train_loader:
    #     # batch_size=1才能显示
    #     # img = torch.squeeze(img)
    #     imshow(img)
    #     label_fromLabelme = torch.squeeze(label_fromLabelme).numpy()
    #     plt.imshow(label_fromLabelme,cmap='gray')
    #     plt.show()
    #     break

    print("using {} images for training".format(train_num))
    print("using {} images for validation".format(val_num))

    device_ids = [2,3]
    model = MsFENet(base_c=64)
    model = nn.DataParallel(model,device_ids=device_ids)
    model.to(device)
    # loss_function = nn.CrossEntropyLoss()
    # params_to_optimize = [p for p in model1_t1.parameters() if p.requires_grad]
    #
    # optimizer = torch.optim.SGD(
    #     params_to_optimize,
    #     lr=lr, momentum=momentum, weight_decay=1e-4
    # )
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = optim.SGD(model.parameters(),lr=lr,momentum=0.9,weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)
    loss_weight = torch.as_tensor([1.0, 1.0], device=device)

    loss_list = []
    save_dir = './model'
    model_num = len(os.listdir('./model'))-1
    train_steps = len(train_loader)

    max_dice = 0
    max_miou = 0
    best_epoch = 0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        scheduler.step()
        for step, data in enumerate(train_bar):
            images, labels = data
            # # ----------------------------------------------------------------
            # label_test = labels[0].cpu().numpy()
            # img_test = np.transpose(images[0].cpu().numpy(), (1, 2, 0))
            # plt.subplot(1, 2, 1)
            # plt.imshow(img_test)
            # plt.subplot(1, 2, 2)
            # plt.imshow(label_test, cmap='gray')
            # plt.show()
            # # ----------------------------------------------------------------
            outputs = model(images.to(device))
            loss = criterion(outputs, labels.to(device),loss_weight=loss_weight,ignore_index=255)
            loss = loss / acc_steps
            loss.backward()
            if (step + 1) % acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            epoch_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.8f}".format(epoch + 1,
                                                                         epochs,
                                                                         loss)
        if (epoch+1) % 20 == 0:
            evaluate(model, val_loader, device, num_classes=2,plot=True)
            torch.save(model.module.state_dict(), os.path.join(save_dir, 'testnet_{}_{}.pkl'.format(model_num + 1,epoch)))

        print("train epoch[{}] running_loss:{:.3f}".format(epoch + 1, epoch_loss))
        confmat,dice = evaluate(model, val_loader, device, num_classes=2)
        confmat.compute()
        if dice > max_dice and confmat.miou > max_miou:
            max_dice = dice
            max_miou = confmat.miou
            best_epoch = epoch
            print('save')
            torch.save(model.module.state_dict(),os.path.join(save_dir, 'testnet_{}_best.pkl'.format(model_num + 1)))

        print(confmat)
        print('dice',dice)
        print('--------------------------------------------')

        loss_list.append(epoch_loss)
    torch.save(model.module.state_dict(), os.path.join(save_dir, 'testnet_{}_final.pkl'.format(model_num + 1)))

    plt.figure()
    plt.plot(loss_list)
    plt.savefig(r'../../loss_curve.jpg')
    # model1_t1.eval()
    print('Finished Training! Best epoch is {}'.format(best_epoch))