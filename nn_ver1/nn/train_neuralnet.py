# coding: utf-8
import sys, os
sys.path.append(os.pardir)

import numpy as np
#from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet
from common.optimizer import *

# データの読み込み
#(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

x_train = np.loadtxt('temp.csv', delimiter=',')
t_train = np.loadtxt('delta_temp.csv', delimiter=',')
x_test = np.loadtxt('temp.csv', delimiter=',')
t_test = np.loadtxt('delta_temp.csv', delimiter=',')

x_train = np.mat(x_train)
t_train = np.mat(t_train)
x_test = np.mat(x_test)
t_test = np.mat(t_test)
#t_train = t_train.T
#t_test = t_test.T

network = TwoLayerNet(input_size=23, hidden_size=100, output_size=23)
optimizer = SGD(lr=0.01)

iters_num = 100
train_size = x_train.shape[0]
#print(train_size) #100
batch_size = 1
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

#iter_per_epoch = max(train_size / batch_size, 1)
iter_per_epoch = 1

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    #print(x_train[batch_mask])
    #print(x_train)
    #print(t_train[2, 0])
    x_batch = x_train[batch_mask, :]
    #t_batch = t_train[batch_mask, 0]
    t_batch = t_train[batch_mask, :]
    print(i)
    #print(x_batch)
    #print(t_batch)
    
    # 勾配
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)
    params = network.params
    optimizer.update(params, grad)

    # 更新
    #for key in ('W1', 'b1', 'W2', 'b2'):
        
        #network.params[key] -= learning_rate * grad[key]
    
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
