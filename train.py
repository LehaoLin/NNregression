#!/usr/bin/python 
# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F     # 激励函数都在这
import torch.nn
import torch.utils.data as Data
# import torchvision      # 数据库模块
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
from torch.autograd import Variable

# 单隐藏层训练模型函数
# ----------------
# 训练函数：  train(x, y, n, epoch_max, lr, mod=relu)
# 使用方法：
# x: 特征属性,例：
# x = [[1,2,3],[2,3,4]]  ([x1,x2]), x1=[...], x2=[...]
# y: label，标签，目标属性
# y = [1,2,3]     y = [...]
#
# x1   x2   y
# 1    2    1
# 2    3    2
# 3    4    3
#
# n 单隐藏层的结点数
# epoch_max: 最大训练次数
# lr: 学习率 学习率低，学的比较慢 参数一种
# mod: 激励函数，默认为 F.relu， 若替换则可使用F.tanh, F.sigmoid 等
# train(x, y, 10, 100, 0.01) -> 
# 返回：生成记录属性平均值和标准差的csv  和 神经网络信息 csv 和 神经网络模型参数

def normalize(x, y):
    numbers = len(x)
    mean_x = []
    div_x = []
    for i in x:
        mean_shit = np.mean(i)
        div_shit = np.std(i, ddof=1)
        mean_x.append(mean_shit)
        div_x.append(div_shit)
  
    mean_y = np.mean(y)
    div_y = np.std(y, ddof=1)
    df0 = pd.DataFrame()
    i = 0
    while i < len(x):
        df0['mean_x'+str(i)] = [mean_x[i]]
        df0['div_x'+str(i)] = [div_x[i]]
        i += 1
    df0['mean_y'] = [mean_y]
    df0['div_y'] = [div_y]
    df0 = df0.reset_index()
    df0.to_csv('mean_div.csv', encoding='utf-8', index = False)
    
    df1 = pd.DataFrame()
    df1 = df1.reset_index()

    k = 0
    while k < len(x):
        for i in x:
            for n in i:
                n = (n-mean_x[k])/div_x[k]
            k += 1
    for i in y:
        i = (i-mean_y)/div_y
    
    p = len(x[0])
    q = 0
    shit = []
    while q < p:
        z = []
        for i in x:
            z.append(i[q])
        shit.append(z)
        q += 1
    df1['input'] = shit
    df1['result'] = y
    x = df1['input']
    y = df1['result']
    print(x)
    print(y)
    y = torch.FloatTensor(df1['result'])

    x =  torch.FloatTensor([x]).squeeze(2)
    x = torch.squeeze(x, dim=1)
    y = torch.unsqueeze(y, dim=1)
    x, y = Variable(x), Variable(y)
    print(x)
    print(y)
    return x, y

# x
# y 
# n: the number of hidden nodes
# epoch_max: >100, int 训练次数
# lr: 0<lr<1, float
def train(x, y, n, epoch_max, lr, mod=F.relu):
    feature = len(x)
    x,  y = normalize(x, y)

    class Net(torch.nn.Module):
        def __init__(self, n_feature, n_hidden, n_output):
            super(Net, self).__init__()
            self.hidden = torch.nn.Linear(n_feature, n_hidden)
            self.predict = torch.nn.Linear(n_hidden, n_output)
        
        def forward(self, x):
            x =  mod(self.hidden(x))
            x = self.predict(x)
            return x

    net = Net(n_feature=feature, n_hidden = n, n_output=1)
    with open("NN.txt","w") as f:
        f.write(str(feature)+','+str(n)+','+str(lr)+','+str(mod.__name__)) 

    

    # optimizer 是训练的工具
    optimizer = torch.optim.SGD(net.parameters(), lr)  # 传入 net 的所有参数, 学习率
    loss_func = torch.nn.MSELoss()       # 预测值和真实值的误差计算公式 (均方差)

    print('开始训练！')
    start = time.time()
    for epoch in range(epoch_max):
        # rank = []
        
        prediction = net(x)  # 喂给 net 训练数据 x, 输出分析值
        # print(prediction)
        loss = loss_func(prediction, y)     # 计算两者的误差
        optimizer.zero_grad()   # 清空上一步的残余更新参数值
        loss.backward()         # 误差反向传播, 计算参数更新值
        optimizer.step()        # 将参数更新值施加到 net 的 parameters 上
        end = time.time()
        if epoch % 10 ==0:
            print('Epoch: ', epoch+1, '| Loss: ', loss.item(),' |训练时间: ',round(end-start,2),'秒')
            print('---------------------------------------------')
            print('保存第',epoch+1,'遍训练模型中...')
            print('---------------------------------------------')
            torch.save(net.state_dict(), 'net.pt') # 保存整个网络参数
            # 训练十次保存一遍

    print('神经网络结束！')
    end = time.time()
    print('总用时: ',end-start,'秒')
    # 训练完成保存一遍
    torch.save(net.state_dict(), 'net.pt') # 保存整个网络参数
    print('保存成功！')



# 测试：
x = [[1,2,3],[2,3,4]]
y = [1,2,3]
train(x, y, 10, 100, 0.01)




