import os

import numpy as np

from Args import TrainArgs, NetProxy
import torch
from torch.utils.data import DataLoader, Dataset
import h5py
from torch.autograd import Variable
from test import test


class TrainSetLoader(Dataset):
    def __init__(self, dataset_dir):
        super(TrainSetLoader, self).__init__()
        self.dataset_dir = dataset_dir
        file_list = os.listdir(dataset_dir)
        item_num = len(file_list)
        self.item_num = item_num

    def __getitem__(self, index):
        dataset_dir = self.dataset_dir
        index = index + 1
        file_name = [dataset_dir + '/%06d' % index + '.h5']
        with h5py.File(file_name[0], 'r') as hf:
            data = np.array(hf.get('data'))
            label = np.array(hf.get('label'))

            data = data.transpose((3, 2, 1, 0))
            label = label.transpose((3, 2, 1, 0))

            data = torch.from_numpy(data)
            label = torch.from_numpy(label)

        return data, label

    def __len__(self):
        return self.item_num


if __name__ == '__main__':
    net_proxy = NetProxy()
    train_args = TrainArgs()
    net_proxy.net.to(train_args.device)
    if train_args.init_kaiming:
        net_proxy.init_kaiming()
    if train_args.load_pretrain:
        best = net_proxy.load()
    else:
        best = 1
    train_set = TrainSetLoader(train_args.trainset_dir)
    train_loader = DataLoader(dataset=train_set,
                              num_workers=train_args.num_works,
                              batch_size=train_args.batch_size,
                              shuffle=True)
    criterion_Loss = torch.nn.L1Loss().to(train_args.device)
    optimizer = torch.optim.Adam([paras for paras in net_proxy.net.parameters() if paras.requires_grad == True],
                                 lr=train_args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=train_args.n_steps, gamma=train_args.gamma)

    print('\n', '***\t' + net_proxy.model_filename[:-8] + '\t' + '***', '\n')

    for idx_epoch in range(train_args.n_epochs):
        for idx_iter, (data, label) in enumerate(train_loader, 0):
            data, label = Variable(data).to(train_args.device), Variable(label).to(train_args.device)
            out = net_proxy.net(data)
            loss = criterion_Loss(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        loss_ave, time_ave = test(net_proxy.net)
        print(idx_epoch,'\t', loss_ave,'\t', best)
        if loss_ave < best:
            best = loss_ave
            net_proxy.save(best=best)
