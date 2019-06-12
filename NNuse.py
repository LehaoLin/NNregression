#!/usr/bin/python 
# -*- coding: utf-8 -*-
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F     # 激励函数都在这
import pandas as pd
import datetime
import time 
import jieba
import numpy as np
import tkinter as tk
import random


# 使用模型
# ----------------
# 使用函数： pred(x)
# 使用方法：
# x: 特征属性,例：
# x = [1,2,3]     ([x1, x2, x3]) float/int
# 返回：预测值 float y


def norm_raw(raw_list):
    f1 = open('mean_div.csv',encoding="utf-8")
    df = pd.read_csv(f1)
    df = df.reset_index()
    list1 = []
    i = 0
    while i < len(raw_list):
        list1.append(raw_list[i]-df['mean_x'+str(i)][0]/(df['div_x'+str(i)][0]))
        i+=1
    return list1

def norm_result(result, raw_list):
    f1 = open('mean_div.csv',encoding="utf-8")
    df = pd.read_csv(f1)
    df = df.reset_index()
    result1 = result * df['div_y'][0]+df['mean_y'][0]
    return result

def pred(con):
    con = norm_raw(con)
    x = torch.FloatTensor(con) 
    with open("NN.txt", "r") as f:    #打开文件
        data = f.readline()   #读取文件
        a  =  data.split(',')
    # print(a)
    feature = int(a[0])
    n = int(a[1])
    lr = float(a[2])
    mod = getattr(F, a[3])   ###

    class Net(torch.nn.Module):
        def __init__(self, n_feature, n_hidden, n_output):
            super(Net, self).__init__()
            self.hidden = torch.nn.Linear(n_feature, n_hidden)
            self.predict = torch.nn.Linear(n_hidden, n_output)
        
        def forward(self, x):
            x =  mod(self.hidden(x))
            x = self.predict(x)
            return x
    # optimizer 是训练的工具
    net = Net(n_feature=feature, n_hidden =n ,n_output=1)
    optimizer = torch.optim.SGD(net.parameters(), lr)  # 传入 net 的所有参数, 学习率
    loss_func = torch.nn.MSELoss()       # 预测值和真实值的误差计算公式 (均方差)
    net.load_state_dict(torch.load('net.pt', map_location = 'cpu'))
    print('模型导入成功！')
    prediction = net(x)
    con.append(1)
    prediction = norm_result(prediction.data.numpy()[0],con)
    return prediction


# # 测试
# a = [1,2]
# print(pred(a))