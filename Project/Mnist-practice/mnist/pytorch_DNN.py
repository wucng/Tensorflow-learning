#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
import torch.optim as optim
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 120)  # 全连接层  通道c 1-->120
        self.fc2 = nn.Linear(120, 84) # 通道c 120-->84
        self.fc3 = nn.Linear(84, 10) # 通道c 84-->10

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
net.cuda()

criterion = nn.CrossEntropyLoss() # use a Classification Cross-Entropy loss
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

digits = load_digits()
# 随机选取75%的数据作为训练样本；其余25%的数据作为测试样本。
# X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
#                                                     test_size=0.25, random_state=33)
# get the inputs
inputs = torch.Tensor(digits.data)
labels=torch.Tensor(digits.target)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    # for i, data in enumerate(trainloader, 0):
    for i in range(2000):
        # wrap them in Variable
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 100 == 0:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')
