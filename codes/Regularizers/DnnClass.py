from __future__ import print_function, division, absolute_import, unicode_literals
import imp
import math
import numpy as np
import torch
from Regularizers.RegularizerClass import RegularizerClass
from dnns.DnCNN import DnCNN
from dnns.UNet import UNet
import torch.nn as nn
from collections import OrderedDict
from tools import utils




class DnnClass(RegularizerClass):
    """
    A Dnn implementation
    """
   
    def __init__(self, model_path, dnn_config=None, dnn_name='dncnn'):
        self.dnn_config = dnn_config
        self.dnn_name = dnn_name

        if self.dnn_name == 'dncnn':
            self.dnn = DnCNN(self.dnn_config).cuda()
        elif self.dnn_name == 'dncnnsp1':
            self.dnn = DnCNNSP1(**self.dnn_config).cuda()
        elif self.dnn_name == 'dncnnsp2':
            self.dnn = DnCNNSP2(**self.dnn_config).cuda()
        elif self.dnn_name == 'unet':
            self.dnn = UNet(**self.dnn_config).cuda()
        self.restore(model_path, dnn_name)

    def restore(self, model_path, dnn_name):
        print(f'\nLoading model at {model_path}')
        if dnn_name == 'dncnnsp2':
            checkpoint = torch.load(model_path)
            self.dnn = nn.DataParallel(self.dnn).cuda()
            self.dnn.load_state_dict(checkpoint,strict=True)
        else:
            checkpoint = torch.load(model_path)['model_state_dict']
            try: 
                name = 'dnn.'
                new_state_dict = OrderedDict({k[len(name):]: v for k, v in checkpoint.items()}) 
                self.dnn.load_state_dict(new_state_dict,strict=True)
            except: 
                dnn_key = []
                name = 'dnn1.'
                for k, v in checkpoint.items():
                    if name in k: 
                        dnn_key.append(k)
                new_state_dict = OrderedDict({k[len(name):]: checkpoint[k] for k in dnn_key}) 
                self.dnn.load_state_dict(new_state_dict,strict=True)
                
    def init(self):
        p = None
        return p
    
    def name(self):
        return self.dnn_name
    

    def eval(self):
        return None

    def denoise(self, x_next, norm, DSClass):
        # normalize data
        Dx, val = DSClass.norm_data(x_next, method=norm)
        # n e h w 2 --> e * n, 2, h, w
        Dx = DSClass.change_dim(Dx, p='to_2dcnn', axes=0)
        # dnn
        Dx = self.dnn(Dx.to(torch.float32))
        #  e * n, 2, h, w --> n e h w 2
        Dx = DSClass.change_dim(Dx, p='from_2dcnn', axes=0)
        # denomralize data: 
        Dx = DSClass.denorm_data(Dx, val=val, method=norm)
        return Dx 

    def prox(self, s, step, pin=None, mu=1, clip=False):
        dataset_dict, dObj_dict, rObj_dict, network_dict, method_dict, runner_dict = utils.get_class_dict()
        DSClass = dataset_dict['csmri']
        if clip:
            s[s<=0] = 0
        else:
            pass
        if len(s.shape) == 5: # s e h w 2
            self.dnn.eval()
            with torch.no_grad():
                batch_pre = self.denoise(s * mu, None, DSClass) / mu
        else:
            print('Incorrect s.shape')
            exit()

        return batch_pre, pin