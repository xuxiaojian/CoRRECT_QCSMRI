import os, datetime
from wsgiref.simple_server import WSGIRequestHandler

# import cv2
import matplotlib.pyplot as plt
import shutil

import numpy as np
import skimage
from skimage.metrics.simple_metrics import mean_squared_error
import torch
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import os
import random
import numpy as np
from scipy.optimize import fminbound


###############################################################################
# Image metrics
###############################################################################
def image_metrics(img_true:torch.Tensor, img_test:torch.Tensor, metrics=['snr'], mode='average', dataset='csmri'):
    # works on torch data
    with torch.no_grad():
        pkg_fun = torch
        # only run on magnitude data
        data_dict = {'true':img_true, 'test':img_test}
        for key, item in data_dict.items():
            if len(item.shape) == 5: # N, E, H, W, M --> N, E, H, W 
                item = torch.view_as_complex(item)
            item = pkg_fun.abs(item)
            N, E, H, W = item.shape
            data_dict[key] = item.view(-1, 1, H, W) 
        img_true = data_dict['true']
        img_test = data_dict['test']

        res = {key: [] for key in metrics}
        for i in range(img_true.shape[0]):
            for name in metrics:
                x =  img_true[i, :]
                xhat = img_test[i, :]
                if name == 'snr':
                    SNR = 20 * pkg_fun.log10(pkg_fun.norm(x.flatten())/pkg_fun.norm(x.flatten()-xhat.flatten()))
                    res[name].append(SNR.cpu().numpy())
                else:
                    print('Metrics not found!')
                    exit(0)
        if mode == 'average':
            return {key: np.mean(res[key]) for key in res.keys()}
        elif mode == 'sum':
            return {key: np.sum(res[key]) for key in res.keys()}
        elif mode == 'multi':
            return res


###############################################################################
# Add noise
###############################################################################
def addwgn_torch(y: torch.Tensor, noise_type='snr', noise_level=0, if_norm=False):
    if noise_level == 0:
        return y
    else:
        complex_flag = False
        if torch.is_complex(y):
            complex_flag = True
            y = torch.view_as_real(y)

        noise_shape = y.size()
        if noise_type == 'snr': # signal domain
            noiseNorm = torch.norm(y.flatten() * 10 ** (-noise_level / 20))
            noise = torch.randn(noise_shape).to(y.device)
            noise = noise / torch.norm(noise.flatten()) * noiseNorm
            min_val = 0
            max_val = 1
        elif noise_type == 'sgm': # image domain
            # normalize
            min_val = torch.min(y)
            y = y - min_val
            max_val = torch.max(y)
            y = y / max_val
            noise = torch.FloatTensor(noise_shape).normal_(mean=0, std=noise_level/255.).to(y.device)

        rec_y = y + noise
        rec_y = rec_y * max_val + min_val
        if complex_flag: rec_y = torch.view_as_complex(rec_y)
    return rec_y

    # noiseNorm = norm(x(:)) * 10^(-inputSnr/20); % compute the desired norm

    # % generate AWGN
    # if(isreal(x))
    #     noise = randn(size(x)); 
    # else
    #     noise = randn(size(x)) + 1j*randn(size(x));
    # end

    # noise = noise/norm(noise(:)) * noiseNorm; % adjust the noise power
    # y = x + noise; % add noise


###############################################################################
# Make folders
###############################################################################

def check_and_mkdir(path):
    if not os.path.exists(path):
        # os.mkdir(path)
        os.makedirs(path)


###############################################################################
# Backup codes
###############################################################################

def copytree_code(src_path, save_path):
    max_code_save = 100
    for i in range(max_code_save):
        code_path = save_path + 'code%d/' % i
        if not os.path.exists(code_path):
            shutil.copytree(src=src_path, dst=code_path)
            break


###############################################################################
# Make tables
###############################################################################

def dict_to_md_table(config: dict):
    # Convert a python dict to markdown table
    info = str()
    for section in config.keys():

        info += '## ' + section + '\n'
        info += '|  Key  |  Value |\n|:----:|:---:|\n'

        # if isinstance(config[section], dict):
        #     for subsection in config[section].keys():
        #         info += '|' + i + '|' + str(config[section][i]) + '|\n'

        for key in config[section].keys():
            info += '|' + key + '|' + str(config[section][key]) + '|\n'

        info += '\n\n'

    return info


###############################################################################
# Init envs
###############################################################################

def init_env(seed_value=0):
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)



###############################################################################
# Optimize tau
###############################################################################

def optimizeTau(x, algoHandle, taurange, maxfun=20):
    # evaluateSNR = lambda x, xhat: 20 * np.log10(
    #     np.linalg.norm(x.flatten('F')) / np.linalg.norm(x.flatten('F') - xhat.flatten('F')))
    # fun = lambda tau: -evaluateSNR(x, algoHandle(tau)[0])

    fun = lambda tau: -image_metrics(x, algoHandle(tau), metrics=['snr'], mode='average')['snr']
    tau = fminbound(fun, taurange[0], taurange[1], xtol=1e-6, maxfun=maxfun, disp=3)
    return tau



###############################################################################
# Init weights
###############################################################################

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025, 0.025)
        nn.init.constant_(m.bias.data, 0.0)



###############################################################################
# Print dictionary
###############################################################################

import json
import pprint
def print_dict(dict, keys='default', print_format='concate'):
    if keys != 'default':
        subdict = {key: dict[key] if key in dict else None for key in keys}
    else:
        subdict = dict

    if print_format == 'singel_line':
        dict_str = pprint.pformat(subdict)
    elif print_format == 'multi_line':
        dict_str = json.dumps(subdict, indent=2, sort_keys=False)
    elif print_format == 'concate':
        dict_str = ''
        for key in subdict: 
            dict_str += f'_{subdict[key]}'
        dict_str = dict_str[1:] #remove the _
    else:
        print(f'print_format [{print_format}] not found !')
        exit(0)

    return dict_str

def pretty(d, indent=0):
    #  Print dictionary 
    for key, value in d.items():
      print('\t' * indent + str(key))
      if isinstance(value, dict):
         pretty(value, indent+1)
      else:
         print('\t' * (indent+1) + str(value))


###############################################################################
# Compute L (torch version)
###############################################################################

def powerIter(A, imgSize, iter=100, tol=1e-6, verbose=False):
    # compute singular value for A'*A
    # A should be a function (lambda:x)

    x = np.random.randn(imgSize[0], imgSize[1])
    x = x / np.linalg.norm(x.flatten('F'))

    lam = 1

    for i in range(iter):
        # apply Ax

        if not torch.is_tensor(x):
            x =  torch.from_numpy(x).float()[None, None, :,:]
            xnext = A(x)

            x = x.squeeze().cpu().numpy()
            xnext = xnext.squeeze().cpu().numpy()


        # xnext' * x / norm(x)^2
        lamNext = np.dot(xnext.flatten('F'), x.flatten('F')) / np.linalg.norm(x.flatten('F')) ** 2
        # only take the real part
        lamNext = lamNext.real

        # normalize xnext 
        xnext = xnext / np.linalg.norm(xnext.flatten('F'))

        # compute relative difference
        relDiff = np.abs(lamNext - lam) / np.abs(lam)

        x = xnext
        lam = lamNext

        # verbose
        if verbose:
            print('[{}/{}] lam = {}, relative Diff = {:0.4f}'.format(i, iter, lam, relDiff))

        # stopping criterion
        if relDiff < tol:
            break

    return lam


def powerIter_MCMRI(A, imgSize, iter=100, tol=1e-6, verbose=False):
    # compute singular value for A'*A
    # A should be a function (lambda:x)

    N, E, H, W, M = imgSize
    x = np.random.randn(N, E, H, W, M)
    x = x / np.linalg.norm(x.flatten('F'))

    lam = 1
    for i in range(iter):
        # apply Ax
        if not torch.is_tensor(x):
            x =  torch.from_numpy(x).float().cuda()
            xnext = A(x)

            x = x.cpu().numpy()
            xnext = xnext.cpu().numpy()
            
        # xnext' * x / norm(x)^2
        lamNext = np.dot(xnext.flatten('F'), x.flatten('F')) / np.linalg.norm(x.flatten('F')) ** 2
        # only take the real part
        lamNext = lamNext.real

        # normalize xnext 
        xnext = xnext / np.linalg.norm(xnext.flatten('F'))

        # compute relative difference
        relDiff = np.abs(lamNext - lam) / np.abs(lam)

        x = xnext
        lam = lamNext

        # verbose
        if verbose:
            print('[{}/{}] lam = {}, relative Diff = {:0.4f}'.format(i, iter, lam, relDiff))

        # stopping criterion
        if relDiff < tol:
            break

    return lam


def ipt_to_kwargs(**kwargs):
    return kwargs


def squeeze_generic(a, axes_to_keep):
    out_s = [s for i,s in enumerate(a.shape) if i in axes_to_keep or s!=1]
    return a.reshape(out_s)


###############################################################################
#  dictionary
###############################################################################

from dataloader.QCSMRIDS import QCSMRIDS

from DataFidelities.CSMRIDF import CSMRIDF
from DataFidelities.LeBIODF import LeBIODF
from Regularizers.DnnClass import DnnClass

from dnns.DnCNN import DnCNN, DnCNNSP1, DnCNNSP2
from dnns.UNet import UNet


from methods.QCSMRI import QCSMRI
from runners.NetRunner import NetRunner


def get_class_dict():
    dataset_dict = {
        'qcsmri':QCSMRIDS,
    }

    dObj_dict = {
        'csmri': CSMRIDF,
        'lebio': LeBIODF,
    }

    rObj_dict = {
        'dnn':DnnClass,
    }

    dnn_dict = {
        'dncnn': DnCNN,
        'unet': UNet,
    }

    method_dict = {
        'qcsmri':QCSMRI,
    }

    runner_dict = {
        'NetRunner': NetRunner,
    }

    return dataset_dict, dObj_dict, rObj_dict, dnn_dict, method_dict, runner_dict



