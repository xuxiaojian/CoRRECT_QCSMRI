# from unicodedata import ucd_3_2_0
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import scipy.io as sio
import numpy as np
import torch
from tools import utils, data_process
# from tools.GLOBAL_INFO import dataset_dict, dObj_dict, rObj_dict, dnn_dict, method_dict, runner_dict


##################################################
# Reproducibility
##################################################
utils.init_env(seed_value=0)

# ##################################################
# # global settings  
# ##################################################
dataset_dict, dObj_dict, rObj_dict, dnn_dict, method_dict, runner_dict = utils.get_class_dict()

##################################################
# load config
##################################################
with open('/export1/project/xiaojianxu/projects/2022-QCSMRI/codes/configs/qcsmri_config.json') as File:
    config = json.load(File)

##################################################
# read config
##################################################
# global settings
gpu_ids = config['settings']['gpu_ids']
project_root = config['settings']['project_root']
code_root = config['settings']['code_root']
exp_root = config['settings']['exp_root']
data_root = config['settings']['data_root']

fold = config['settings']['fold']
problem_name = config['settings']['problem_name']
method_name = config['settings']['method_name']
description = config['settings']['description']

# subsections
problem_config = config['problems'][problem_name]
method_config = config['methods'][method_name]

##################################################
# init the gpu usages
##################################################
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
##################################################
# get forward model
##################################################
print('\n[Forward model]:')
A_config = problem_config["A_config"]
A_file_path = f'{data_root}/Mask/msk_{tuple(A_config["sigSize"])}_f{fold}'
A = data_process.read_mri_files(A_file_path, file_type='msk', opt_type='tensor')
A_flag = f'f{fold}'
print(A_flag)

##################################################
# get method config
##################################################
rule = method_config['rule']
dnn1_name = method_config['dnn1_name']
dnn2_name = method_config['dnn2_name']

##################################################
# get dObj 
##################################################
dObj1 = dObj_dict['csmri'](A)
dnn1 = dnn_dict[dnn1_name](config['networks'][dnn1_name])

te = torch.from_numpy(np.array(range(4, 44, 4)) / 1e3).float()
dObj2 = dObj_dict['lebio'](te)
dnn2 = dnn_dict[dnn2_name](config['networks'][dnn2_name])

##################################################
# get method 
##################################################
print('\n[Method]:')
method = method_dict[method_name](dObj1, dObj2, dnn1, dnn2, config)
method_flag = method.flag()
print(method)

##################################################
# get runner
######################################f############
print('\n[Runner]:')
Runner = runner_dict['NetRunner'](method, device, config)

##################################################
# run 
##################################################
print('\n[Experiment runnning]:')
mode = method_config['mode']
if mode == 'train':
    print('\n[Dataloader]:')
    train_with_test = method_config['train']['train_with_test']
    train_dataset = dataset_dict[problem_name](A, config, device, 'train')
    valid_dataset = dataset_dict[problem_name](A, config, device, 'valid')
    test_dataset = dataset_dict[problem_name](A, config, device,'test') if train_with_test else None
    dataset_flag = train_dataset.flag()

    exp_flag = f'{A_flag}_{dataset_flag}_{method_flag}{description}'
    Runner.train(train_dataset=train_dataset, valid_dataset=valid_dataset, test_dataset=test_dataset, exp_folder=exp_flag)
elif mode == 'test' :
    subj_list = config['problems'][problem_name][mode]['data_name']
    for subj in subj_list:
        config_subj  = config
        config_subj['problems'][problem_name][mode]['data_name'] = [subj]
        test_dataset = dataset_dict[problem_name](A, config_subj, device, 'test')
        dataset_flag = test_dataset.flag()
        Runner = runner_dict['NetRunner'](method, device, config)
        
        if method_name  == 'pnp': 
            Runner.pnp(test_dataset=test_dataset)
        else: 
            Runner.test(test_dataset=test_dataset)



