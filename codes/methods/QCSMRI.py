from cgitb import reset
from math import gamma
from sys import prefix
import torch
import torch.nn as nn
from tools import utils
import numpy as np
import scipy.io as sio
from runners import iterAlgs
from collections import OrderedDict


class QCSMRI(nn.Module):
    def __init__(self, dObj1, dObj2, dnn1, dnn2, config):
        super(QCSMRI, self).__init__()
        self.dObj1 = dObj1
        self.dObj2 = dObj2
        self.dnn1 = dnn1
        self.dnn2 = dnn2

        self.config = config
        self.name = 'qcsmri'

        # parameters: mu-scaling, alpha-, gamma-stepsize, epsilon-threshold
        method_config = self.config['methods'][self.name]
        rule = method_config['rule']
        # self.weight = nn.Parameter(torch.tensor(method_config['train_paras'][rule]['weight'][0], dtype=torch.float32), requires_grad=method_config['train_paras'][rule]['weight'][1])
        self.gamma = nn.Parameter(torch.tensor(method_config['train_paras'][rule]['gamma'][0], dtype=torch.float32), requires_grad=method_config['train_paras'][rule]['gamma'][1])
        if method_config['rule'] == 'red':
            self.tau = nn.Parameter(torch.tensor(method_config['train_paras'][rule]['tau'][0], dtype=torch.float32), requires_grad=method_config['train_paras'][rule]['tau'][1])
        elif method_config['rule'] == 'pnp':
            self.mu = nn.Parameter(torch.tensor(method_config['train_paras'][rule]['mu'][0], dtype=torch.float32), requires_grad=method_config['train_paras'][rule]['mu'][1])
            self.alpha = nn.Parameter(torch.tensor(method_config['train_paras'][rule]['alpha'][0], dtype=torch.float32), requires_grad=method_config['train_paras'][rule]['alpha'][1])

    def flag(self):
        return self.config['methods'][self.name]['rule']

    def warmup(self, warmup_path):
        for name, path in warmup_path.items():
            dnn_prefix = 'dnn.'
            if name == 'dnn1' and path:
                checkpoint = torch.load(path)['model_state_dict']
                try: self.dnn1.load_state_dict(checkpoint,strict=True)
                except: 
                    dnn_key = []
                    method_config = self.config['methods'][self.name]
                    rule = method_config['rule']
                    for k, v in checkpoint.items():
                        if k == 'gamma': self.gamma.copy_(v) #self.gamma = nn.Parameter(torch.tensor(v, dtype=torch.float32), requires_grad=method_config['train_paras'][rule]['gamma'][1])
                        elif k == 'tau': self.tau.copy_(v) #self.tau = nn.Parameter(torch.tensor(v, dtype=torch.float32), requires_grad=method_config['train_paras'][rule]['tau'][1])
                        elif k == 'mu':  self.mu.copy_(v) #self.mu =  nn.Parameter(torch.tensor(v, dtype=torch.float32), requires_grad=method_config['train_paras'][rule]['mu'][1])
                        elif k == 'alpha':self.alpha.copy_(v) # self.alpha =  nn.Parameter(torch.tensor(v, dtype=torch.float32), requires_grad=method_config['train_paras'][rule]['alpha'][1])
                        elif dnn_prefix in k: dnn_key.append(k)
                    new_state_dict = OrderedDict({k[len(dnn_prefix):]: checkpoint[k] for k in dnn_key}) 
                    self.dnn1.load_state_dict(new_state_dict,strict=True)
            if name == 'dnn2' and path :
                checkpoint = torch.load(path)['model_state_dict']
                try: self.dnn2.load_state_dict(checkpoint,strict=True)
                except: 
                    new_state_dict = OrderedDict({k[len(dnn_prefix):]: v for k, v in checkpoint.items()}) 
                    self.dnn2.load_state_dict(new_state_dict,strict=True)

    def denoise(self, x_next, norm, DSClass, rule):
        # normalize data
        Dx, val = DSClass.norm_data(x_next, method=norm)
        # n e h w 2 --> e * n, 2, h, w
        Dx = DSClass.change_dim(Dx, p='to_2dcnn', axes=0)
        # dnn
        if rule == 'pnp': Dx = self.dnn1(Dx.to(torch.float32) * self.mu)/self.mu
        else: Dx = self.dnn1(Dx.to(torch.float32))
        #  e * n, 2, h, w --> n e h w 2
        Dx = DSClass.change_dim(Dx, p='from_2dcnn', axes=0)
        # denomralize data: 
        Dx = DSClass.denorm_data(Dx, val=val, method=norm)
        return Dx 

    def unroll(self, x_init, y, csm):
        # Dataset Obj
        dataset_dict, dObj_dict, rObj_dict, network_dict, method_dict, runner_dict = utils.get_class_dict()
        DSClass = dataset_dict[self.config['settings']['problem_name']]
        # configs
        method_config = self.config['methods'][self.name]
        rule = method_config['rule']
        num_iter = method_config['num_iter']
        norm = method_config['norm']
         # run
        x_next = x_init
        for it in range(num_iter):
            # gradient of dObj     
            grad_d = self.dObj1.grad(x_next, y, csm)  
            # update
            if  rule == 'red':
                Dx = self.denoise(x_next, norm, DSClass, rule)
                x_next = x_next - self.gamma * (grad_d + self.tau * (x_next - Dx))
            elif rule == 'pnp': 
                x1_next = x_next - self.gamma * grad_d
                x2_next = self.denoise(x1_next, norm, DSClass, rule)
                x_next = self.alpha * x1_next + (1 - self.alpha) * x2_next
            else:
                print('Rule not found!')
                exit(0)
        return x_next

    def forward(self, x_init, x_true, scale, y, csm, ft, bmsk, gdt_no_bmsk): 
        # run unroll: s,e, h, w, 2 --> s, e, h, w, 2
        mGRE = self.unroll(x_init, y, csm)
        # change dimension: s, e, h, w, 2 --> s, e, h, w
        mGRE_tmp = torch.abs(torch.view_as_complex(mGRE))
        # run lebio: s, e, h, w --> s, 2, h, w
        S0R2s = self.dnn2(mGRE_tmp)
        return (S0R2s,mGRE)



