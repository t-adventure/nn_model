# coding: utf-8
import sys, os
sys.path.append(os.pardir)

import numpy as np
from two_layer_net import TwoLayerNet
from common.optimizer import *
from setting_data import *

# データの読み込み
delta = 300 / 20 # delta_data sampling rate
x_train, t_train = setting_data(delta)

x_test = x_train
t_test = t_train

x_train = np.mat(x_train)
t_train = np.mat(t_train)
x_test = np.mat(x_test)
t_test = np.mat(t_test)

input_size = x_train.shape[1]
output_size = t_train.shape[1]
hidden_size = 200

network = TwoLayerNet(input_size, hidden_size, output_size)
optimizer = SGD(lr=0.00000000001)

#iters_num = min(x_train.shape[0], t_train.shape[0])
iters_num = 20

train_size = x_train.shape[0]
batch_size = 1

train_loss_list = []
train_acc_list = []
test_acc_list = []

#iter_per_epoch = max(train_size / batch_size, 1)
iter_per_epoch = 1

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask, :]
    t_batch = t_train[batch_mask, :]
    print(i)
    #print(x_batch)
    #print(t_batch)

    # 勾配
    grad = network.gradient(x_batch, t_batch)
    params = network.params
    optimizer.update(params, grad)

    # 更新
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)

#np.savetxt('train_acc_list.csv', train_acc_list, delimiter=", ")
np.savetxt('train_loss_list.csv', train_loss_list, delimiter=", ")
np.savetxt('W1.csv', network.params['W1'], delimiter=", ")
np.savetxt('b1.csv', network.params['b1'], delimiter=", ")
np.savetxt('W2.csv', network.params['W2'], delimiter=", ")
np.savetxt('b2.csv', network.params['b2'], delimiter=", ")
