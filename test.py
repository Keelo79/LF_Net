import torch
import numpy as np
import h5py
import cv2
import os
import time
from Args import TrainArgs, NetProxy

train_args = TrainArgs()


def test(net):
    ave_score = []
    ave_time = []
    criterion_Loss = torch.nn.L1Loss().to(train_args.device)
    file_list = os.listdir(train_args.testset_dir)

    for filename in file_list:
        with h5py.File(train_args.testset_dir + '/' + filename, 'r') as hf:
            data = np.array(hf.get('data'))
            label = np.array(hf.get('label'))
            data = data.transpose((3, 2, 1, 0))
            label = label.transpose((3, 2, 1, 0))
            data = torch.from_numpy(data).to(train_args.device)
            label = torch.from_numpy(label).to(train_args.device)
            data = data.unsqueeze(0)
            label = label.unsqueeze(0)
            with torch.no_grad():
                start = time.time()
                data = net(data)
                end = time.time()
                ave_time.append(end - start)
            ave_score.append(criterion_Loss(data, label).data.cpu())
    loss_ave = np.array(ave_score).mean()
    time_ave = np.array(ave_time).mean()
    return loss_ave, time_ave


if __name__ == '__main__':
    net_proxy = NetProxy()
    net_proxy.net.to(train_args.device)
    net_proxy.load()
    loss_ave, time_ave = test(net_proxy.net)
    print('average_score=', loss_ave)
    print('average_time=', time_ave)
