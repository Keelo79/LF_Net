import torch
import numpy as np
import h5py
import cv2
import os
import time
import matplotlib.pyplot as plt
from Args import TrainArgs, NetProxy

mode = 'cv'
assert mode == 'plt' or mode == 'cv'
name = '000040.h5'

train_args = TrainArgs()
criterion_Loss = torch.nn.L1Loss().to(train_args.device)
net_proxy = NetProxy()
net_proxy.net.to(train_args.device)
net_proxy.load()

with h5py.File('./' + train_args.testset_dir + '/' + name, 'r') as hf:
    data = hf.get('data')
    label = hf.get('label')

    data, label = np.array(data), np.array(label)
    data = data.transpose((3, 2, 1, 0))
    label = label.transpose((3, 2, 1, 0))
    data, label = torch.from_numpy(data).to(train_args.device), torch.from_numpy(label).to(train_args.device)
    data = data.unsqueeze(0)
    label = label.unsqueeze(0)
    with torch.no_grad():
        start = time.time()
        sr = net_proxy.net(data)
        end = time.time()
        print('time=', end - start)
        print('score=', criterion_Loss(sr, label).data.to(train_args.device))
        data_show = np.array(data[0, 0, 0, :, :].cpu())
        label_show = np.array(label[0, 0, 0, :, :].cpu())
        sr_show = np.array(sr[0, 0, 0, :, :].cpu())
        if mode == 'plt':
            plt.imshow(data_show)
            plt.show()
            plt.imshow(label_show)
            plt.show()
            plt.imshow(sr_show)
            plt.show()
        if mode == 'cv':
            cv2.imshow('data', data_show)
            cv2.imshow('label', label_show)
            cv2.imshow('sr', sr_show)
            cv2.waitKey()
