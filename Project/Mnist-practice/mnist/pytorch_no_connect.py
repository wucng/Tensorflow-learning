# -*- coding: UTF-8 -*-

"""
数据：mnist  data.shape 28x28X1；lables.shape 0~9 这里使用非one-hot标签（Tensorflow转成One-hot标签）
data 从Nx28X28x1[N h w c] 转成 Nx1x28x28[N c h w]  Tensorflow使用[N h w c]格式
模型：卷积神经网络 
模型结构：卷积核 3x3 strides 1x1 ；池化核 2x2 

舍弃全连接层
输入矩阵[N 1 28 28]  通过卷积池化 [N 10 1 1] 在展成[N 10](取代全连接层)（10 表示10个类别）



改成自己的模型： 数据传入接口（改写）
"""

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')  # 训练 每批训练样本数
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)') # 测试 每批测试样本数
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)') # 迭代次数
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')    # 学习效率（超参数）
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,  #是否无cuda
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',  # 随机因子
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status') # 每个多少次输出训练状态信息
args = parser.parse_args()  # 整合所有参数
args.cuda = not args.no_cuda and torch.cuda.is_available() # 设置成cuda 模式

torch.manual_seed(args.seed) # 设置随机因子
if args.cuda:
    torch.cuda.manual_seed(args.seed) #随机确定cuda（有多块GPU时）


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {} #如果cuda存在，设置工作节点为1，
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True, # ../data 为下载目录，train=True 训练数据，download=True 下载
                   transform=transforms.Compose([  # 组合
                       transforms.ToTensor(), # 转成Tensor
                       transforms.Normalize((0.1307,), (0.3081,)) # 均值为0.1307，方差为0.3081 进行数据正则化，mnist数据通道为1，所以均值和方差第一维为1
                       # 如果通道为3，应该是[0.1307,0.1307,0.1307]  [0.3081,0.3081,0.3081] 数值可以自行定义
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs) # shuffle=True 随机打乱数据
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([ # train=False 对应测试数据
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)


class Net(nn.Module): # 自定义网络模型
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3) # 通道数 1-->10 (mnist 通道数 1) 卷积核大小 5x5
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3) # 通道数 10-->20 (mnist 通道数 1) 卷积核大小 5x5
        self.conv2_drop = nn.Dropout2d()  # dropout层 防止过拟合

        # self.conv2 = nn.Conv2d(20, 20, kernel_size=3)
        self.conv3 = nn.Conv2d(20, 10, kernel_size=3)

        # self.fc1 = nn.Linear(320, 50)  # 全连接层 320-->50 （320=20x4x4） 20为卷积池化后的通道c，4为卷机池化后的h，w
                                        # （pytorch 卷积输入的图像矩阵格式为[N c h w]） 需从[N h w c]转成[N c h w]
        # self.fc2 = nn.Linear(50, 10)  # 全连接层 50-->10 （10个类别）

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2)) # 先进行最大池化，2x2的核，再进行relu
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2)) # dropout ，池化，relu
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv3(x)), 2))
        x = x.view(-1, 10*1*1)  # 将每张图片展开成一行，一行代表一张图像
        # x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training) # dropout training 训练 进行 dropout
        # x = self.fc2(x)
        return F.log_softmax(x) # softmax激活函数

model = Net()
if args.cuda:
    model.cuda() #设置GPU模式（模型gpu模式）

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum) #(梯度下降)优化器

def train(epoch):
    model.train() # 训练
    for batch_idx, (data, target) in enumerate(train_loader):
        # print("label.size:",target.size(),"label:",target[0:10]) # ===>torch.Size([64]) 采用的不是one-hot
        # print(data.size())   # ===>torch.Size([64, 1, 28, 28])  已从[64,28,28,1] 转到 [64, 1, 28, 28]
        # input('===========')
        if args.cuda:
            data, target = data.cuda(), target.cuda()  #GPU模式
        data, target = Variable(data), Variable(target) # Tensor-->variable
        optimizer.zero_grad() # 参数初始化（梯度初始化）
        output = model(data) # 预测值
        loss = F.nll_loss(output, target) # loss
        loss.backward() # loss 反馈 相当于 optimizer.minst(loss)
        optimizer.step() # 每步训练优化器
        if batch_idx % args.log_interval == 0: # 每隔一定步数打印信息
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test(epoch):
    model.eval() # 测试
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum() # cpu() 将数据从gpu转到cpu

    test_loss = test_loss
    test_loss /= len(test_loader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
