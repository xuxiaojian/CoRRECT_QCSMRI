import torch
import torch.nn as nn
import torch.nn.functional as F

activation_fn = {
    'relu': lambda: nn.ReLU(inplace=True),
    'lrelu': lambda: nn.LeakyReLU(inplace=True),
    'prelu': lambda: nn.PReLU()
}


'''
###########################################################################
# UNet
###########################################################################
'''
class ConvBnActivation(nn.Module):
    def __init__(self, in_channels, out_channels, dimension=2, times=1, is_bn=False, activation='relu', kernel_size=3):
        super().__init__()

        if dimension == 3:
            conv_fn = lambda in_c: torch.nn.Conv3d(in_channels=in_c,
                                                   out_channels=out_channels,
                                                   kernel_size=kernel_size,
                                                   padding=kernel_size // 2
                                                   )
            bn_fn = lambda: torch.nn.BatchNorm3d(out_channels)

        elif dimension == 2:
            conv_fn = lambda in_c: torch.nn.Conv2d(in_channels=in_c,
                                                   out_channels=out_channels,
                                                   kernel_size=kernel_size,
                                                   padding=kernel_size // 2
                                                   )
            bn_fn = lambda: torch.nn.BatchNorm2d(out_channels)
        else:
            raise ValueError()

        layers = []
        for i in range(times):
            if i == 0:
                layers.append(conv_fn(in_channels))
            else:
                layers.append(conv_fn(out_channels))

            if is_bn:
                layers.append(bn_fn())

            if activation is not None:
                layers.append(activation_fn[activation]())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ConvtranBnActivation(nn.Module):
    def __init__(self, in_channels, out_channels, dimension=2, is_bn=False, activation='relu', kernel_size=3):
        self.is_bn = is_bn
        super().__init__()
        if dimension == 3:
            conv_fn = lambda: torch.nn.ConvTranspose3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=(1, 2, 2),
                padding=kernel_size // 2,
                output_padding=(0, 1, 1)
            )
            bn_fn = lambda: torch.nn.BatchNorm3d(out_channels)

        elif dimension == 2:
            conv_fn = lambda: torch.nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=kernel_size // 2,
                output_padding=1
            )
            bn_fn = lambda: torch.nn.BatchNorm2d(out_channels)
        else:
            raise ValueError()

        self.net1 = conv_fn()
        if self.is_bn:
            self.net2 = bn_fn()
        self.net3 = activation_fn[activation]()

    def forward(self, x):
        ret = self.net1(x)
        if self.is_bn:
            ret = self.net2(ret)

        ret = self.net3(ret)

        return ret


class UNet(nn.Module):
    def __init__(self, net_config):
        super(UNet, self).__init__()
        self.name = 'unet'

        conv_times = 3
        dimension = net_config['dimension']
        i_nc = net_config['ic']
        o_nc = net_config['oc']
        f_root = net_config['features']
        is_bn = net_config['is_bn']
        is_sn = net_config['is_sn']
        activation = net_config['activation']

        self.up_down_times = net_config['depth']
        self.is_residual = net_config['is_res']
        self.depth = self.up_down_times
        self.flag = f'{self.name}_{self.depth}'

        if dimension == 2:
            self.down_sample = nn.MaxPool2d(2)
        elif dimension == 3:
            self.down_sample = nn.MaxPool3d((1, 2, 2))
        else:
            raise ValueError()

        self.conv_in = ConvBnActivation(
            in_channels=i_nc,
            out_channels=f_root,
            is_bn=is_bn,
            activation=activation,
            dimension=dimension)

        self.conv_out = ConvBnActivation(
            in_channels=f_root,
            out_channels=o_nc,
            kernel_size=1,
            dimension=dimension,
            times=1,
            is_bn=False,
            activation=None
        )

        self.bottom = ConvBnActivation(
            in_channels=f_root * (2 ** (self.up_down_times - 1)),
            out_channels=f_root * (2 ** self.up_down_times),
            times=conv_times, is_bn=is_bn, activation=activation, dimension=dimension)

        self.down_list = nn.ModuleList([
                                           ConvBnActivation(
                                               in_channels=f_root * 1,
                                               out_channels=f_root * 1,
                                               times=conv_times, is_bn=is_bn, activation=activation,
                                               dimension=dimension)
                                       ] + [
                                           ConvBnActivation(
                                               in_channels=f_root * (2 ** i),
                                               out_channels=f_root * (2 ** (i + 1)),
                                               times=conv_times, is_bn=is_bn, activation=activation,
                                               dimension=dimension)
                                           for i in range(self.up_down_times - 1)
                                       ])

        self.up_conv_list = nn.ModuleList([
            ConvBnActivation(
                in_channels=f_root * (2 ** (self.up_down_times - i)),
                out_channels=f_root * (2 ** (self.up_down_times - i - 1)),
                times=conv_times, is_bn=is_bn, activation=activation, dimension=dimension)
            for i in range(self.up_down_times)
        ])

        self.up_conv_tran_list = nn.ModuleList([
            ConvtranBnActivation(
                in_channels=f_root * (2 ** (self.up_down_times - i)),
                out_channels=f_root * (2 ** (self.up_down_times - i - 1)),
                is_bn=is_bn, activation=activation, dimension=dimension)
            for i in range(self.up_down_times)
        ])

    def forward(self, x):

        input_ = x

        x = self.conv_in(x)

        skip_layers = []
        for i in range(self.up_down_times):
            x = self.down_list[i](x)

            skip_layers.append(x)
            x = self.down_sample(x)

        x = self.bottom(x)

        for i in range(self.up_down_times):
            x = self.up_conv_tran_list[i](x)
            x = torch.cat([x, skip_layers[self.up_down_times - i - 1]], 1)
            x = self.up_conv_list[i](x)

        x = self.conv_out(x)

        ret = input_ - x if self.is_residual else x

        return ret
