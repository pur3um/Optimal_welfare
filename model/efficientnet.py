# import package
# https://deep-learning-study.tistory.com/563
# https://supermemi.tistory.com/139
# https://supermemi.tistory.com/132
import pandas as pd
from PIL import Image
import cv2

# model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch import optim

# dataset and transformation
from torchvision import datasets  # dataset 여기서는 사용하지 않음
# https://pseudo-lab.github.io/Tutorial-Book/chapters/object-detection/Ch3-preprocessing.html
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
# from dataset.ImbalancedSampler import getImbalancedSampler ##
from torchvision import models
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

# display images
from torchvision import utils
import matplotlib.pyplot as plt

# utils
import numpy as np
from torchsummary import summary
import time
import copy, pickle


def data_load(opt):
    """
    opt : train, val
    img data를 로드하고 tensor로 변경하여 제공, torch로 만들어야함
    """
    file_path = f"../../data_preprocessing/{opt}"
    # csv_path = "../../rsna-breast-cancer-detection/train.csv"
    # csv_file = pd.read_csv(csv_path)
    # 특정 cancer 값만 return

    """# check img
    someimg = os.listdir(file_path)[0]

    img = cv2.imread(os.path.join(file_path, someimg))
    # 4 channel을 유지하고 싶다면 cv2.IMREAD_UNCHANGED 사용
    # img = cv2.imread(os.path.join(train_path, someimg), cv2.IMREAD_UNCHANGED)

    img_array = np.array(img)  # (983, 512, 3), Height * Width * Channel
    # toTensor()를 통해 C * H * W로 변경

    # torchvision.transforms.ToTensor
    tf_toTensor = transforms.ToTensor()
    img_RGB_tensor_frrom_ndarray = tf_toTensor(img_array)"""

    # check data : tensor로 잘 변경되었는지 확인
    # print(img_RGB_tensor_frrom_ndarray)
    # print(img_RGB_tensor_frrom_ndarray.size())  # 3 * 983 * 512
    # print(img_RGB_tensor_frrom_ndarray.min(), img_RGB_tensor_frrom_ndarray.max())  # 0 ~ 1 사이의 값

    data_name = []
    # label = []
    for png_name in os.listdir(file_path):
        data_name.append(os.path.join(file_path, png_name))
        # for idx, value in csv_file.iterrows():
        #     if int(png_name.split("_")[1][:-4]) == value["image_id"]:
        #         print(png_name)
        #         label.append(value["cancer"])
        #         break
    with open("../../data_preprocessing/label.pickle", "rb") as fr:
        label = pickle.load(fr)
    # 이번만 저장해두기, npy, pickle 중에 더 적합한 포맷을 선정
    # with open("../../data_preprocessing/label.pickle", "wb") as f:
    #     pickle.dump(label, f)
    # data save npy or other format
    return data_name, label

# 데이터셋 구축
class RsnaDataset(Dataset):
    def __init__(self):
        self.transform = transforms.ToTensor()
        self.file_path_list, self.y = data_load("train")
        # y = pd.read_csv(self.csv_path)
        # self.y = y["cancer"].values

    def __len__(self):
        return len(self.file_path_list)

    def __getitem__(self, idx):
        png = cv2.imread(self.file_path_list[idx])  # cv2.IMREAD_GRAYSCALE
        png_array = np.array(png)
        # label =
        tt = self.transform(png_array)  # img-ary -> Tensor


        return tt, self.y[idx]  # y를 idx로 접근하기 좀 그렇다면, dict 형태로 해서

# model 구축
# Swish activation function: 활성화 함수 정의
class Swish(nn.Module):  # = silu
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)

# Effnet은 SE Block을 사용힘
class SEBlock(nn.Module):
    def __init__(self, in_channels, r=4):
        super().__init__()

        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels * r),
            Swish(),
            nn.Linear(in_channels * r, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.squeeze(x)
        x = x.view(x.size(0), -1)
        x = self.excitation(x)
        x = x.view(x.size(0), x.size(1), 1, 1)
        return x

class MBConv(nn.Module):
    expand = 6
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, se_scale=4, p=0.5):
        super().__init__()
        # first MBConv is not using stochastic depth
        self.p = torch.tensor(p).float() if (in_channels == out_channels) else torch.tensor(1).float()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * MBConv.expand, 1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(in_channels * MBConv.expand, momentum=0.99, eps=1e-3),
            Swish(),
            nn.Conv2d(in_channels * MBConv.expand, in_channels * MBConv.expand, kernel_size=kernel_size,
                      stride=1, padding=kernel_size//2, bias=False, groups=in_channels*MBConv.expand),
            nn.BatchNorm2d(in_channels * MBConv.expand, momentum=0.99, eps=1e-3),
            Swish()
        )

        self.se = SEBlock(in_channels * MBConv.expand, se_scale)

        self.project = nn.Sequential(
            nn.Conv2d(in_channels*MBConv.expand, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.99, eps=1e-3)
        )

        self.shortcut = (stride == 1) and (in_channels == out_channels)

    def forward(self, x):
        # stochastic depth
        if self.training:
            if not torch.bernoulli(self.p):
                return x

        x_shortcut = x
        x_residual = self.residual(x)
        x_se = self.se(x_residual)

        x = x_se * x_residual
        x = self.project(x)

        if self.shortcut:
            x= x_shortcut + x

        return x

class SepConv(nn.Module):
    """ SepConv 입니다. MBConv와의 차이점은 expand=1인 것과 2개의 layer로 구성되어 있습니다. MBConv는 3개의 layer로 구정되어 있고, expand=6 입니다. """
    expand = 1
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, se_scale=4, p=0.5):
        super().__init__()
        # first SepConv is not using stochastic depth
        self.p = torch.tensor(p).float() if (in_channels == out_channels) else torch.tensor(1).float()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels * SepConv.expand, in_channels * SepConv.expand, kernel_size=kernel_size,
                      stride=1, padding=kernel_size//2, bias=False, groups=in_channels*SepConv.expand),
            nn.BatchNorm2d(in_channels * SepConv.expand, momentum=0.99, eps=1e-3),
            Swish()
        )

        self.se = SEBlock(in_channels * SepConv.expand, se_scale)

        self.project = nn.Sequential(
            nn.Conv2d(in_channels*SepConv.expand, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.99, eps=1e-3)
        )

        self.shortcut = (stride == 1) and (in_channels == out_channels)

    def forward(self, x):
        # stochastic depth
        if self.training:
            if not torch.bernoulli(self.p):
                return x

        x_shortcut = x
        x_residual = self.residual(x)
        x_se = self.se(x_residual)

        x = x_se * x_residual
        x = self.project(x)

        if self.shortcut:
            x= x_shortcut + x

        return x

# EfficientNet 정의
class EfficientNet(nn.Module):
    def __init__(self, num_classes=10, width_coef=1., depth_coef=1., scale=1., dropout=0.2, se_scale=4, stochastic_depth=False, p=0.5):
        super().__init__()
        channels = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
        repeats = [1, 2, 2, 3, 3, 4, 1]
        strides = [1, 2, 2, 2, 1, 2, 1]
        kernel_size = [3, 3, 5, 3, 5, 5, 3]
        depth = depth_coef
        width = width_coef

        channels = [int(x*width) for x in channels]
        repeats = [int(x*depth) for x in repeats]

        # stochastic depth
        if stochastic_depth:
            self.p = p
            self.step = (1 - 0.5) / (sum(repeats) - 1)
        else:
            self.p = 1
            self.step = 0


        # efficient net
        self.upsample = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False)

        self.stage1 = nn.Sequential(
            nn.Conv2d(3, channels[0],3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[0], momentum=0.99, eps=1e-3)
        )

        self.stage2 = self._make_Block(SepConv, repeats[0], channels[0], channels[1], kernel_size[0], strides[0], se_scale)

        self.stage3 = self._make_Block(MBConv, repeats[1], channels[1], channels[2], kernel_size[1], strides[1], se_scale)

        self.stage4 = self._make_Block(MBConv, repeats[2], channels[2], channels[3], kernel_size[2], strides[2], se_scale)

        self.stage5 = self._make_Block(MBConv, repeats[3], channels[3], channels[4], kernel_size[3], strides[3], se_scale)

        self.stage6 = self._make_Block(MBConv, repeats[4], channels[4], channels[5], kernel_size[4], strides[4], se_scale)

        self.stage7 = self._make_Block(MBConv, repeats[5], channels[5], channels[6], kernel_size[5], strides[5], se_scale)

        self.stage8 = self._make_Block(MBConv, repeats[6], channels[6], channels[7], kernel_size[6], strides[6], se_scale)

        self.stage9 = nn.Sequential(
            nn.Conv2d(channels[7], channels[8], 1, stride=1, bias=False),
            nn.BatchNorm2d(channels[8], momentum=0.99, eps=1e-3),
            Swish()
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(channels[8], num_classes)

    def forward(self, x):
        x = self.upsample(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.stage8(x)
        x = self.stage9(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.linear(x)
        return x


    def _make_Block(self, block, repeats, in_channels, out_channels, kernel_size, stride, se_scale):
        strides = [stride] + [1] * (repeats - 1)
        layers = []
        for stride in strides:
            layers.append(block(in_channels, out_channels, kernel_size, stride, se_scale, self.p))
            in_channels = out_channels
            self.p -= self.step

        return nn.Sequential(*layers)


def efficientnet_b0(num_classes=10):
    return EfficientNet(num_classes=num_classes, width_coef=1.0, depth_coef=1.0, scale=1.0,dropout=0.2, se_scale=4)

def efficientnet_b1(num_classes=10):
    return EfficientNet(num_classes=num_classes, width_coef=1.0, depth_coef=1.1, scale=240/224, dropout=0.2, se_scale=4)

def efficientnet_b2(num_classes=10):
    return EfficientNet(num_classes=num_classes, width_coef=1.1, depth_coef=1.2, scale=260/224., dropout=0.3, se_scale=4)

def efficientnet_b3(num_classes=10):
    return EfficientNet(num_classes=num_classes, width_coef=1.2, depth_coef=1.4, scale=300/224, dropout=0.3, se_scale=4)

def efficientnet_b4(num_classes=10):
    return EfficientNet(num_classes=num_classes, width_coef=1.4, depth_coef=1.8, scale=380/224, dropout=0.4, se_scale=4)

def efficientnet_b5(num_classes=10):
    return EfficientNet(num_classes=num_classes, width_coef=1.6, depth_coef=2.2, scale=456/224, dropout=0.4, se_scale=4)

def efficientnet_b6(num_classes=10):
    return EfficientNet(num_classes=num_classes, width_coef=1.8, depth_coef=2.6, scale=528/224, dropout=0.5, se_scale=4)

def efficientnet_b7(num_classes=10):
    return EfficientNet(num_classes=num_classes, width_coef=2.0, depth_coef=3.1, scale=600/224, dropout=0.5, se_scale=4)

# print model summary
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = efficientnet_b3().to(device)
# 983 * 512
# summary(model, (3, 983, 512), device=device.type)

# def img(path):
#     transform = transforms.ToTensor()
#     png = cv2.imread(path)  # cv2.IMREAD_GRAYSCALE
#     png_array = np.array(png)
#     tt = transform(png_array)  # img-ary -> Tensor
#     return tt
#
# test = img("../../data_preprocessing/val/106_2018825992.png")
#
# # https://stackoverflow.com/questions/49941426/attributeerror-collections-ordereddict-object-has-no-attribute-eval
# # model 호출
# model.load_state_dict(torch.load("./models/weights2.pt"))
# print(model.eval())

#### 학습에 필요한 함수 정의
# define loss function, optimizer, lr_scheduler
loss_func = nn.CrossEntropyLoss(reduction='sum')
opt = optim.Adam(model.parameters(), lr=0.01)



lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=10)


# get current lr
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']


# calculate the metric per mini-batch
def metric_batch(output, target):
    pred = output.argmax(1, keepdim=True)
    corrects = pred.eq(target.view_as(pred)).sum().item()
    return corrects


# calculate the loss per mini-batch
def loss_batch(loss_func, output, target, opt=None):
    loss_b = loss_func(output, target)
    metric_b = metric_batch(output, target)

    if opt is not None:
        opt.zero_grad()
        loss_b.backward()
        opt.step()

    return loss_b.item(), metric_b


# calculate the loss per epochs
def loss_epoch(model, loss_func, dataset_dl, sanity_check=False, opt=None):
    running_loss = 0.0
    running_metric = 0.0
    len_data = len(dataset_dl.dataset)

    for xb, yb in dataset_dl:
        xb = xb.to(device)
        yb = yb.to(device)
        output = model(xb)

        loss_b, metric_b = loss_batch(loss_func, output, yb, opt)

        running_loss += loss_b

        if metric_b is not None:
            running_metric += metric_b

        if sanity_check is True:
            break

    loss = running_loss / len_data
    metric = running_metric / len_data
    return loss, metric


# function to start training
def train_val(model, params):
    num_epochs = params['num_epochs']
    loss_func = params['loss_func']
    opt = params['optimizer']
    train_dl = params['train_dl']
    val_dl = params['val_dl']
    sanity_check = params['sanity_check']
    lr_scheduler = params['lr_scheduler']
    path2weights = params['path2weights']

    loss_history = {'train': [], 'val': []}
    metric_history = {'train': [], 'val': []}

    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    start_time = time.time()

    for epoch in range(num_epochs):
        current_lr = get_lr(opt)
        print('Epoch {}/{}, current lr= {}'.format(epoch, num_epochs - 1, current_lr))

        model.train()
        train_loss, train_metric = loss_epoch(model, loss_func, train_dl, sanity_check, opt)
        loss_history['train'].append(train_loss)
        metric_history['train'].append(train_metric)

        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model, loss_func, val_dl, sanity_check)
        loss_history['val'].append(val_loss)
        metric_history['val'].append(val_metric)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), path2weights)
            torch.save(model, "./models/model.pt")
            print('Copied best model weights!')

        lr_scheduler.step(val_loss)
        if current_lr != get_lr(opt):
            print('Loading best model weights!')
            model.load_state_dict(best_model_wts)

        print('train loss: %.6f, val loss: %.6f, accuracy: %.2f, time: %.4f min' % (
        train_loss, val_loss, 100 * val_metric, (time.time() - start_time) / 60))
        print('-' * 10)

    model.load_state_dict(best_model_wts)
    return model, loss_history, metric_history


bsz4myDesktop = {
    's': 10,
    'm': 16,
    'l': 18
}


dataset = RsnaDataset()
dataset_size = len(dataset)
train_size = int(dataset_size * 0.9)
validation_size = dataset_size - train_size
train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])
# sampler = getImbalancedSampler(train_dataset, train_dataset.indices)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=bsz4myDesktop["s"],
    num_workers=2,
    # sampler=sampler
)
validation_loader = DataLoader(dataset=validation_dataset, batch_size=bsz4myDesktop["s"], num_workers=8,
                               shuffle=False)

# train_dl = DataLoader()
# define the training parameters
params_train = {
    'num_epochs':100,
    'optimizer': opt,
    'loss_func': loss_func,
    'train_dl': train_loader,
    'val_dl': validation_loader,
    'sanity_check':False,
    'lr_scheduler':lr_scheduler,
    'path2weights':'./models/weights2.pt',
}

# check the directory to save weights.pt
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except:  # OSerror
        print('Error')

createFolder('./models')



# check
if __name__ == '__main__':
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     x = torch.randn(3, 3, 224, 224).to(device)
#     model = efficientnet_b0().to(device)
#     output = model(x)
#     print('output size:', output.size())
#
#     # print model summary
#     model = efficientnet_b7().to(device)
#     summary(model, (3, 224, 224), device=device.type)
    # 학습
    model, loss_hist, metric_hist = train_val(model, params_train)

    #  loss-accuracy progress를 출력합니다.
    num_epochs = params_train['num_epochs']

    # Plot train-val loss
    plt.title('Train-Val Loss')
    plt.plot(range(1, num_epochs + 1), loss_hist['train'], label='train')
    plt.plot(range(1, num_epochs + 1), loss_hist['val'], label='val')
    plt.ylabel('Loss')
    plt.xlabel('Training Epochs')
    plt.legend()
    plt.show()

    # plot train-val accuracy
    plt.title('Train-Val Accuracy')
    plt.plot(range(1, num_epochs + 1), metric_hist['train'], label='train')
    plt.plot(range(1, num_epochs + 1), metric_hist['val'], label='val')
    plt.ylabel('Accuracy')
    plt.xlabel('Training Epochs')
    plt.legend()
    plt.show()