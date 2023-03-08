#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import pickle
from glob import glob
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
import paddle
import paddle.nn as nn
from paddle import fluid
from KalmanFilter import Kalman_filter
from DeepStateSpace import DeepSS

#place = fluid.CUDAPlace(0) if paddle.device.get_cudnn_version else fluid.CPUPlace()
paddle.device.set_device('cpu')

data_path='Train\\Train'   #存放数据的路径
pkl_files = glob(data_path+'/*.pkl')

ind_pkl_files = []   #正样本
ood_pkl_files = []   #负样本
for each_path in tqdm(pkl_files):
    pic = open(each_path,'rb')
    this_pkl_file= pickle.load(pic)
    if this_pkl_file[1]['label'] == '00':
        ind_pkl_files.append(each_path)
    else:
        ood_pkl_files.append(each_path)

random.seed(0)
random.shuffle(ind_pkl_files)
random.shuffle(ood_pkl_files)

train_pkl_files = ind_pkl_files[:6000]
train_pkl_files.extend(ood_pkl_files[:2000])
test_pkl_files = ind_pkl_files[6000:]
test_pkl_files.extend(ood_pkl_files[2000:])

#import DataDivision.data_div
#train_pkl_files, test_pkl_files = data_div(ind_pkl_files, ood_pkl_files)

def load_data(pkl_list,label=True):
    X = []
    y = []

    for  each_pkl in pkl_list:
        pic = open(each_pkl,'rb')
        item= pickle.load(pic)
        X.append(item[0][:,0:7])
        if label:
            y.append(int(item[1]['label'][0]))

    X = np.vstack(X)
    if label:
        y = np.vstack(y)
    return X, y

X_train, y_train = load_data(train_pkl_files)
X_test, y_test = load_data(test_pkl_files)

_mean = np.mean(X_train, axis=0)
_std = np.std(X_train, axis=0)
X_train = (X_train - _mean) / (_std + 1e-4)
X_test = (X_test - _mean) / (_std + 1e-4)

X_train, X_test = X_train.reshape(-1, 256, 7), X_test.reshape(-1, 256, 7)

X_train, y_train = paddle.to_tensor(X_train, dtype="float32"), paddle.to_tensor(y_train, dtype="float32")


class PyODDataset(paddle.io.Dataset):
    def __init__(self, X, y):
        super(PyODDataset, self).__init__()
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        sample = self.X[idx, :]
        label = self.y[idx,:]
        return paddle.to_tensor(sample,dtype="float32"), paddle.to_tensor(label,dtype="int32")


###开始训练
def train(X_train, y_train, X_test, y_test, input_size=7, lr=0.05, batch_size=512, epochs=20, cosine_factor=1):
    train_dataset = PyODDataset(X_train, y_train)
    train_loader = paddle.io.DataLoader(train_dataset, places=paddle.CPUPlace(), batch_size=batch_size, shuffle=True, num_workers=0)
    
    deepss = DeepSS(input_size=input_size)
    kf_loss = Kalman_filter()
    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=lr, T_max=epochs,  eta_min=cosine_factor * lr,verbose=False)
    optim = paddle.optimizer.Adam(learning_rate=scheduler, parameters=deepss.parameters())
    mse_loss = nn.MSELoss()
    #deepss_state_dict = paddle.load('./deepss.pkl')
    #deepss.set_state_dict(deepss_state_dict)

    for i in range(epochs):
        for j, (X_train, y_train) in enumerate(train_loader()):
            params, y_pred = deepss(X_train)
            
            F, H = params[:, :49].reshape([-1, 256, 7, 7]), params[:, 49:98].reshape([-1, 256, 7, 7])
            b, Q, R = params[:, 98:105].reshape([-1, 256, 7, 1]), params[:, 105:112].reshape([-1, 256, 7, 1]), params[:, 112:119].reshape([-1, 256, 7, 1])
            Q = paddle.matmul(Q, paddle.transpose(Q, [0, 1, 3, 2]))
            R = paddle.matmul(R, paddle.transpose(R, [0, 1, 3, 2]))
            L = params[::256, 119:126].reshape([-1, 1, 7, 1])
            P = params[::256, 126:133].reshape([-1, 1, 7, 1])
            eye_tensor = paddle.broadcast_to(paddle.eye(7), shape=[batch_size, 1, 7, 7])
            P = eye_tensor * P
            z = X_train.reshape([batch_size, 256, 7, 1])
            
            _, _, loss_state = kf_loss.kalman_filtering(F, H, b, Q, R, L, P, z)
            loss_results = mse_loss(y_pred, y_train.astype('float32'))
            auc = evaluate(y_train, y_pred)

            loss = loss_results + loss_state
            loss = paddle.mean(loss)
            loss.backward()
            optim.minimize(loss)
            deepss.clear_gradients()

        if i%2==0:
            print(i,'total_loss:',loss, 'auc:' auc)
            print('loss_state:', loss_state, 'loss_results:', loss)
            print('#'*20)

    with paddle.no_grad():
        _, y_pred = deepss(X_test)
        auc = evaluate(y_test, y_pred)
        print('acc_val:', acc)
        
    return deepss

def evaluate(label,score):
    fpr, tpr, thresholds = roc_curve(label, score, pos_label=1)
    AUC = auc(fpr, tpr)
    return AUC

if __name__ == '__main__':
    deepss = train(X_train, y_train, X_test, y_test, input_size=7, lr=0.02, epochs=20, cosine_factor=1)
    save_path1 = 'deepss.pkl'
    paddle.save(deepss.state_dict(),save_path1)


