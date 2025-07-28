
import torch.nn as nn

'''
###########################################################################
# DnCNN
###########################################################################
'''
class DnCNN(nn.Module):
    def __init__(self, net_config):
        super(DnCNN, self).__init__()
        self.name = 'dncnn' 
        ic = net_config['ic']
        oc = net_config['oc']
        depth = net_config['depth']
        features = net_config['features']
        kernel_size = net_config['kernel_size']
        is_bn = net_config['is_bn']
        is_sn = net_config['is_sn']
        self.is_res = net_config['is_res']
        padding = (kernel_size - 1) // 2

        layers = []
        layers.append(nn.Conv2d(in_channels=ic, out_channels=features, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=True))
            if is_bn:
                layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=oc, kernel_size=kernel_size, padding=padding, bias=True))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        if self.is_res:
            out = x - out
        return out

