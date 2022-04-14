import torch
from nets.DimNet_Minus import DimNet_Minus
from nets.DimNet_Plus import DimNet_Plus


class TrainArgs():
    def __init__(self):
        self.device = 'cuda:0'
        self.batch_size = 2
        self.lr = 5e-4
        self.n_epochs = 500
        self.n_steps = 50
        self.gamma = 0.7
        self.num_works = 1
        self.trainset_dir = 'data_train'
        self.testset_dir = 'data_test' # 训练时建议将其更改为 data_train
        self.load_pretrain = False
        self.init_kaiming = False


class NetProxy():
    def __init__(self):
        self.ang_res = 5
        self.upscale_factor = 4
        self.receptive_field = 9
        self.activation = 'None' # None ReLu Sigmoid

        # DimNet_Plus
        self.n_block = 2
        self.channels = 2

        self.net_name = 'DimNet_Minus'
        assert self.net_name == 'DimNet_Minus' or self.net_name == 'DimNet_Plus'

        if self.net_name == 'DimNet_Minus':
            self.net = DimNet_Minus(ang_res=self.ang_res,
                                    receptive_field=self.receptive_field,
                                    upscale_factor=self.upscale_factor,
                                    activation=self.activation)

        if self.net_name == 'DimNet_Plus':
            self.net = DimNet_Plus(n_block=self.n_block,
                                   ang_res=self.ang_res,
                                   receptive_field=self.receptive_field,
                                   upscale_factor=self.upscale_factor,
                                   channels=self.channels,
                                   activation=self.activation)

        self.model_filename = self.net.model_filename()

    def init_kaiming(self):
        def weights_init_xavier(m):
            classname = m.__class__.__name__
            if classname.find('Conv3') != -1:
                torch.nn.init.kaiming_uniform_(m.weight.data)

        self.net.apply(weights_init_xavier)

    def load(self):
        weight = torch.load('models/' + self.model_filename)
        self.net.load_state_dict(weight['net'])
        best = weight['best']
        return best

    def save(self, best):
        state = {'net': self.net.state_dict(), 'best': best}
        torch.save(state, 'models/' + self.model_filename)


if __name__ == '__main__':
    train_args = TrainArgs()
    net_proxy = NetProxy()
    net_proxy.net = net_proxy.net.to(train_args.device)
    tmp = torch.zeros((1, 5, 5, 80, 80)).to(train_args.device)
    out = net_proxy.net(tmp)
    print(out.size())
