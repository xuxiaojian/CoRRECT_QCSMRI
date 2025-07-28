import os, time, datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.utils
from torch.utils.tensorboard import SummaryWriter
import scipy.io as sio
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
from tools import utils, transcript, data_process
import csv
from tabulate import tabulate
from collections import OrderedDict
from tqdm import tqdm
from runners import iterAlgs
import h5py


class NetRunner(nn.Module):
    def __init__(self, model, device, config):
        super(NetRunner, self).__init__()
        self.model = model
        self.config = config
        self.device = device
        dataset_dict, dObj_dict, rObj_dict, network_dict, method_dict, runner_dict = utils.get_class_dict()
        self.dataset_name = self.config['settings']['problem_name']
        self.method_name  = self.config['settings']['method_name']
        self.DSClass = dataset_dict[self.dataset_name]
        self.MEClass = method_dict[self.method_name]
        print('NetRunner initialized')

    # =================================================================================================================
    # Train
    # =================================================================================================================
    # @ profile
    def train(self, train_dataset=None, valid_dataset=None, test_dataset=None, exp_folder=''):
        #############################################################
        # read settings
        #############################################################
        # setting config
        gpu_ids = self.config['settings']['gpu_ids']
        code_root = self.config['settings']['code_root']
        exp_root = self.config['settings']['exp_root']

        problem_name = self.config['settings']['problem_name']
        method_name = self.config['settings']['method_name']
        desp = self.config['settings']['description']

        # method configg
        method_config = self.config['methods'][method_name]
        num_workers = method_config['num_workers']
        rule = method_config['rule']
        train_paras = method_config['train_paras'][rule]

        # method - train -config
        train_config = method_config['train']
        warmup_path = train_config['warmup_path']
        restore_path = train_config['restore_path']
        save_folder = train_config['save_folder']
        keep_training = train_config['keep_training']
        train_with_test = train_config['train_with_test']
        

        train_metrics = train_config['metrics']
        train_batch_size = train_config['batch_size'] * torch.cuda.device_count()

        lr = train_config['lr']
        train_epoch = train_config['train_epoch']
        save_epoch = train_config['save_epoch']
        log_epoch = train_config['log_epoch']

        loss_type = train_config['loss_type']
        grad_clip_bar = train_config['grad_clip_bar']

        factor = train_config['factor']
        patience = train_config['patience']

        # method-valid-config
        valid_config = method_config['valid']
        valid_metrics = valid_config['metrics']
        valid_batch_size = valid_config['batch_size'] * torch.cuda.device_count()


        #############################################################
        # init settings
        #############################################################
        # define folders
        if save_folder:
            exp_folder = save_folder
        else:
            exp_folder = f'{exp_root}/{method_name}/{str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))}_{exp_folder}'
        model_folder = f'{exp_folder}/models'
        valid_folder = f'{exp_folder}/valid'

        # make folders
        utils.check_and_mkdir(model_folder)
        utils.check_and_mkdir(valid_folder)

        # init a logfile
        transcript.start(exp_folder + '/train_logfile.log', mode='a')

        # init a tensorboard
        writer = SummaryWriter(exp_folder)

        # init loss
        if loss_type == 'l2':
            criterion = nn.MSELoss(reduction='sum').to(self.device)
        # if self.method_name == 'qcsmri':
        #     mri_weight = nn.Parameter(torch.tensor(train_paras['mri_weight'][0], dtype=torch.float32), requires_grad=train_paras['mri_weight'][1]).to(self.device)


        # init optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        if factor and patience:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=factor, patience=patience, verbose=True)

        #############################################################
        # init data
        #############################################################
        # init code data
        utils.copytree_code(code_root, exp_folder + '/')

        # init tensorboard data
        writer.add_text('init/config', utils.dict_to_md_table(self.config))

        # init model data
        self.model = torch.nn.DataParallel(self.model).to(self.device)

        if warmup_path:
            with torch.no_grad():
                self.model.module.warmup(warmup_path)
                print(f'Warmup loaded from {warmup_path}')

        if restore_path:
            save_dict = torch.load(restore_path)
            self.model.module.load_state_dict(save_dict['model_state_dict'])
            optimizer.load_state_dict(save_dict['optimizer_state_dict']) # xiaojian

        if keep_training:
            best_metrics = {key: save_dict['best_metrics'][key] for key in valid_metrics}
            start_epoch = save_dict['curr_metrics'][valid_metrics[0]]['epoch'] + 1
            print(f'Continue training: Load {restore_path} at epoch = {start_epoch}')
        else:
            start_epoch = 0
            best_metrics = {key:{'value': 0, 'epoch': 0} for key in valid_metrics}
            print(f'Start new training: new model at {start_epoch:d}/{train_epoch:d}')

        #############################################################
        # init dataset
        #############################################################
        # init dataloader
        train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
        valid_dataloader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
        
        # init samples: ipt, gdt, mea
        train_idx_smp, valid_idx_smp = train_dataset.sample_index, valid_dataset.sample_index
        train_ipt_smp, train_gdt_smp,  *train_args_smp = (sample.to(self.device) for sample in next(iter(DataLoader(Subset(train_dataset, train_idx_smp), batch_size=len(train_idx_smp)))))
        valid_ipt_smp, valid_gdt_smp,  *valid_args_smp = (sample.to(self.device) for sample in next(iter(DataLoader(Subset(valid_dataset, valid_idx_smp), batch_size=len(valid_idx_smp)))))

        # make grid images: ipt, gdt 
        train_ipt_smp_grid = torchvision.utils.make_grid(self.DSClass.change_dim(train_ipt_smp, *train_args_smp, p='tb_grid'), nrow=5, normalize=False, scale_each=False)
        train_gdt_smp_grid = torchvision.utils.make_grid(self.DSClass.change_dim(train_gdt_smp, *train_args_smp, p='tb_grid'), nrow=5, normalize=False, scale_each=False)

        valid_ipt_smp_grid = torchvision.utils.make_grid(self.DSClass.change_dim(valid_ipt_smp, *valid_args_smp, p='tb_grid'), nrow=5, normalize=False, scale_each=False)
        valid_gdt_smp_grid = torchvision.utils.make_grid(self.DSClass.change_dim(valid_gdt_smp, *valid_args_smp, p='tb_grid'), nrow=5, normalize=False, scale_each=False)

        # init the ipt metrics
        train_ipt_smp_metrics = utils.image_metrics(train_gdt_smp, train_ipt_smp, metrics=train_metrics, dataset=self.dataset_name)
        valid_ipt_smp_metrics = utils.image_metrics(valid_gdt_smp, valid_ipt_smp, metrics=valid_metrics, dataset=self.dataset_name)

        # tensorboard: train valid
        writer.add_histogram('init/train/hist_ipt', train_ipt_smp)
        writer.add_histogram('init/train/hist_gdt', train_gdt_smp)
        if self.dataset_name != 'qcsmri': writer.add_histogram('init/train/hist_diff', train_ipt_smp - train_gdt_smp)
        writer.add_image('init/train/img_gdt', train_gdt_smp_grid, dataformats='CHW')
        writer.add_image('init/train/img_ipt:{}'.format({key: np.round(train_ipt_smp_metrics[key], 2) for key in train_metrics}), train_ipt_smp_grid, global_step=0, dataformats='CHW')

        writer.add_histogram('init/valid/hist_ipt', valid_ipt_smp)
        writer.add_histogram('init/valid/hist_gdt', valid_gdt_smp)
        if self.dataset_name != 'qcsmri': writer.add_histogram('init/valid/hist_diff', valid_ipt_smp - valid_gdt_smp)
        writer.add_image('init/valid/img_gdt', valid_gdt_smp_grid, dataformats='CHW')
        writer.add_image('init/valid/img_ipt:{}'.format({key: np.round(valid_ipt_smp_metrics[key], 2) for key in valid_metrics}), valid_ipt_smp_grid, global_step=0, dataformats='CHW')

        if train_with_test:
            test_dataloader = DataLoader(test_dataset, batch_size=valid_batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
            test_idx_smp = test_dataset.sample_index
            test_ipt_smp, test_gdt_smp,  *test_args_smp = (sample.to(self.device) for sample in next(iter(DataLoader(Subset(test_dataset, test_idx_smp), batch_size=len(test_idx_smp)))))
            test_ipt_smp_grid = torchvision.utils.make_grid(self.DSClass.change_dim(test_ipt_smp, *test_args_smp, p='tb_grid'), nrow=5, normalize=False, scale_each=False)
            test_gdt_smp_grid = torchvision.utils.make_grid(self.DSClass.change_dim(test_gdt_smp, *test_args_smp, p='tb_grid'), nrow=5, normalize=False, scale_each=False)
            test_ipt_smp_metrics = utils.image_metrics(test_gdt_smp, test_ipt_smp, metrics=valid_metrics, dataset=self.dataset_name)
            writer.add_image('init/test/img_gdt', test_gdt_smp_grid, dataformats='CHW')
            writer.add_image('init/test/img_ipt:{}'.format({key: np.round(test_ipt_smp_metrics[key], 2) for key in valid_metrics}), test_ipt_smp_grid, global_step=0, dataformats='CHW')

        #############################################################
        # Start experiment
        #############################################################
        start_t = time.time()
        for epoch in range(start_epoch, train_epoch):
            # record the training process
            train_records_epoch = {
                'loss': [],
                'ipt_metrics': {key: [] for key in train_metrics},  # record for all iterations at this epoch
                'pre_metrics': {key: [] for key in train_metrics},
            }

            #############################################################
            # Train
            #############################################################
            self.model.train()
            for counter, batch in enumerate(train_dataloader, 0):

                # get global step
                global_step = epoch * len(train_dataloader) + counter

                # zero cache from last step
                self.model.zero_grad()
                optimizer.zero_grad()

                # perform the forward pass
                ipt, gdt, *args = (data.to(self.device) for data in batch)
                pre = self.model(ipt, gdt, *args)
                pre_comp = self.DSClass.change_dim(pre, *args, p='to_loss')
                
                # compute loss
                if self.method_name == 'qcsmri':
                    gdt_no_mask = args[-1]
                    loss_mri = criterion(pre[1], gdt_no_mask) # s e h w 2
                    loss_qmri = criterion(pre_comp,torch.abs(torch.view_as_complex(gdt)))
                    print(f'loss_mri = {loss_mri}, loss_qmri = {loss_qmri}')
                    # bmsk = args[-2]
                    # loss_diff = criterion(pre_comp, torch.abs(torch.view_as_complex(pre[1])) * bmsk)
                    loss_iter = 0.5 * loss_mri + 0.5 * loss_qmri
                elif self.method_name == 'leself':
                    bmsk = args[-2]
                    loss_iter = criterion(pre_comp, torch.abs(torch.view_as_complex(pre[1])) * bmsk)
                else:
                    loss_iter = criterion(pre_comp,gdt)

                # perform the backward pass
                loss_iter.backward()

                # clip gradient
                if grad_clip_bar:
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=grad_clip_bar)

                # update the network
                optimizer.step()

                # summurize the metrics                
                ipt_metrics_iter = utils.image_metrics(gdt, ipt, metrics=train_metrics, dataset=self.dataset_name)
                pre_metrics_iter = utils.image_metrics(gdt, pre_comp, metrics=train_metrics, dataset=self.dataset_name)

                writer.add_scalar('iter/train/loss', loss_iter.item(), global_step)
                for key in train_metrics:
                    writer.add_scalar('iter/train/{}_pre'.format(key), pre_metrics_iter[key], global_step)

                # update the record
                train_records_epoch['loss'].append(loss_iter.item())
                for key in train_metrics:
                    train_records_epoch['ipt_metrics'][key].append(ipt_metrics_iter[key])
                    train_records_epoch['pre_metrics'][key].append(pre_metrics_iter[key])

                # update the logging
                if counter % log_epoch == 0:
                    print(f'[Epoch: {epoch:04d}/{train_epoch:04d}]',
                          f'(Batch: [{counter:02d}] / [{len(train_dataloader)}])',
                          f'Loss: {loss_iter.item():.2f}',
                          f'|| ipt: {str({key: np.round(ipt_metrics_iter[key], 2) for key in train_metrics})}',
                          f'|| pre: {str({key: np.round(pre_metrics_iter[key], 2) for key in train_metrics})}')

            # train pre & tra record for the epoch
            # compute the average
            loss_avg = np.mean(train_records_epoch['loss'])
            ipt_metrics_avg = {key: np.mean(train_records_epoch['ipt_metrics'][key]) for key in train_metrics}
            pre_metrics_avg = {key: np.mean(train_records_epoch['pre_metrics'][key]) for key in train_metrics}

            # record the para
            writer.add_scalar('paras/train/lr', optimizer.param_groups[0]['lr'], epoch)
            for para_name in train_paras:
                if para_name == 'gamma': para = self.model.module.gamma
                elif para_name == 'tau': para = self.model.module.tau
                elif para_name == 'alpha': para = self.model.module.alpha
                elif para_name == 'mu': para = self.model.module.mu
                elif para_name == 'weight': para= self.model.module.weight
                writer.add_scalar(f'paras/train/{para_name}', para.item(), epoch)
            
            # summurize the gradient
            grads_min = []
            grads_max = []
            for param in optimizer.param_groups[0]['params']:
                if param.grad is not None:
                    grads_min.append(torch.min(param.grad))
                    grads_max.append(torch.max(param.grad))
            grads_min = torch.min(torch.stack(grads_min, 0))
            grads_max = torch.max(torch.stack(grads_max, 0))
            writer.add_scalar('paras/train/grad', grads_min.item(), epoch)
            writer.add_scalar('paras/train/grad', grads_max.item(), epoch)

            # update tensorboard
            writer.add_scalar('epoch/train/loss', loss_avg, epoch)
            for key in train_metrics:
                writer.add_scalar(f'epoch/train/{key}_pre', pre_metrics_avg[key], epoch)

            # print statistics
            if epoch % log_epoch == 0:
                print(f'\t[Train][Epoch: {epoch:04d}/{train_epoch:04d}]',
                      f'|| Loss: {loss_avg:3.2f}'
                      f'|| ipt: {str({key: np.round(ipt_metrics_avg[key], 2) for key in train_metrics})}',
                      f'|| pre: {str({key: np.round(pre_metrics_avg[key], 2) for key in train_metrics})}')

            #############################################################
            # Valid
            #############################################################
            valid_results = {'pre': [],
                             'ipt_metrics': [],
                             'pre_metrics': []}
            valid_records_epoch = {
                'ipt_metrics': {key: [] for key in valid_metrics},  # record for all images at this epoch
                'pre_metrics': {key: [] for key in valid_metrics},
            }
            self.model.eval()
            with torch.no_grad():
                for counter, batch in enumerate(valid_dataloader, 0):
                    # perform the forward pass
                    ipt, gdt, *args = (data.to(self.device) for data in batch)
                    pre = self.model(ipt, gdt, *args)
                    pre_comp = self.DSClass.change_dim(pre, *args, p='to_loss')

                    # results
                    ipt_metrics = utils.image_metrics(gdt, ipt, metrics=valid_metrics, dataset=self.dataset_name, mode='multi')
                    if self.method_name == 'qcsmri':
                        gdt_no_mask = args[-1]
                        pre_metrics_mri = utils.image_metrics(gdt_no_mask, pre[1], metrics=valid_metrics, dataset=self.dataset_name, mode='multi')
                        pre_metrics_qmri = utils.image_metrics(gdt, pre_comp, metrics=valid_metrics, dataset=self.dataset_name, mode='multi')
                        pre_metrics = {key: pre_metrics_mri[key] + pre_metrics_qmri[key] for key in valid_metrics}
                        print(f'pre_metrics_mri = {str({key: np.round(pre_metrics_mri[key], 2) for key in pre_metrics_mri})}')
                        print(f'pre_metrics_qmri = {str({key: np.round(pre_metrics_qmri[key], 2) for key in pre_metrics_qmri})}')
                    else:
                        pre_metrics = utils.image_metrics(gdt, pre_comp, metrics=valid_metrics, dataset=self.dataset_name, mode='multi')

                    # update records
                    for key in valid_metrics:
                        valid_records_epoch['ipt_metrics'][key].append(ipt_metrics[key])
                        valid_records_epoch['pre_metrics'][key].append(pre_metrics[key])

                    #store data
                    valid_results['pre'].append(self.DSClass.change_dim(pre, *args, p='to_save'))
                    valid_results['ipt_metrics'].append(ipt_metrics)
                    valid_results['pre_metrics'].append(pre_metrics)

                # reform the resulst
                valid_results['pre'] = np.concatenate(valid_results['pre'], axis=0)

                # compute the average
                ipt_metrics_avg = {key: np.mean(valid_records_epoch['ipt_metrics'][key]) for key in valid_metrics}
                pre_metrics_avg = {key: np.mean(valid_records_epoch['pre_metrics'][key]) for key in valid_metrics}
                curr_metrics = {key: {'value': pre_metrics_avg[key], 'epoch': epoch} for key in valid_metrics}

                # update tensorboard
                for key in valid_metrics:
                    writer.add_scalar(f'epoch/valid/{key}_pre', pre_metrics_avg[key], epoch)

                # print statistic
                print(f'\t[Valid][Epoch: {epoch:04d}/{train_epoch:04d}]',
                      f'|| ipt: {str({key: np.round(ipt_metrics_avg[key], 2) for key in valid_metrics})}',
                      f'|| pre: {str({key: np.round(pre_metrics_avg[key], 2) for key in valid_metrics})}')

                # update tensorboard: text
                model_info_eval = ""
                for key in valid_metrics:
                    if curr_metrics[key]['value'] > best_metrics[key]['value']:
                        model_info_eval += f"[{key}] improve in epoch {epoch}, from {best_metrics[key]['value']:.2f} to {curr_metrics[key]['value']:.2f} \n\n"
                        best_metrics[key]['value'] = curr_metrics[key]['value']
                        best_metrics[key]['epoch'] = curr_metrics[key]['epoch']
                    else:
                        model_info_eval += f"{key} doesn't improve, the best is in epoch {best_metrics[key]['epoch']} with value {best_metrics[key]['value']:.2f} \n\n"
                writer.add_text('model/save_info', model_info_eval, epoch)

                # valid info
                epoch_info = ""
                epoch_info += f"[ipt]{str({key: np.round(valid_records_epoch['ipt_metrics'][key], 2) for key in valid_metrics})}\n\n"
                epoch_info += f"[pre]{str({key: np.round(valid_records_epoch['pre_metrics'][key], 2) for key in valid_metrics})}\n\n"
                writer.add_text('valid/valid_info', epoch_info, epoch)

                #############################################################
                # Train sample for tensorboard
                #############################################################
                # update tensorboard: samples-image, scaler, hist
                pre = self.model(train_ipt_smp, train_gdt_smp, *train_args_smp)
                pre_grid = torchvision.utils.make_grid(self.DSClass.change_dim(pre, *train_args_smp, p='tb_grid'), nrow=5, normalize=False, scale_each=False)
                writer.add_image('sample/train/img_pre', pre_grid, epoch, dataformats='CHW')
                # pre_comp = self.DSClass.change_dim(pre, *args, p='to_loss')
                # writer.add_histogram('sample/train/hist_pre', pre_comp)
                # writer.add_histogram('sample/train/hist_diff', pre_comp - gdt)

                #############################################################
                # Valid sample for tensorboard
                #############################################################
                pre = self.model(valid_ipt_smp, valid_gdt_smp, *valid_args_smp)
                pre_grid = torchvision.utils.make_grid(self.DSClass.change_dim(pre, *valid_args_smp, p='tb_grid'), nrow=5, normalize=False, scale_each=False)
                writer.add_image('sample/valid/img_pre', pre_grid, epoch, dataformats='CHW')
                # pre_comp = self.DSClass.change_dim(pre, *args, p='to_loss')
                # writer.add_histogram('sample/valid/hist_pre', pre_comp)
                # writer.add_histogram('sample/valid/hist_diff', pre_comp - gdt)

                #############################################################
                # Test
                #############################################################
                if train_with_test:
                    pre = self.model(test_ipt_smp, test_gdt_smp, *test_args_smp)
                    pre_grid = torchvision.utils.make_grid(self.DSClass.change_dim(pre, *test_args_smp, p='tb_grid'), nrow=5, normalize=False, scale_each=False)
                    writer.add_image('sample/test/img_pre', pre_grid, epoch, dataformats='CHW')

            #############################################################
            # Global update
            #############################################################
            # update the learning rate
            if factor and patience:
                scheduler.step(curr_metrics['snr']['value'])
            #############################################################
            # Save model/data
            #############################################################
            save_dict = {
                'config': self.config,
                'model_state_dict': self.model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'curr_metrics': curr_metrics,
                'best_metrics': best_metrics
            }

            # save the latest
            sio.savemat(f'{valid_folder}/latest.mat', valid_results)
            torch.save(save_dict, f'{model_folder}/latest.pth')
            print(f'\t[Save the latest] Model/Data saved to {model_folder}/latest.pth')

            # save milestone
            if epoch % save_epoch == 0:
                sio.savemat(f'{valid_folder}/epoch_{epoch}.mat', valid_results)
                torch.save(save_dict, f'{model_folder}/epoch_{epoch}.pth')
                print(f'\t[Save the epoch] Model/Data saved to {model_folder}/epoch_{epoch}.pth')

            # save the best
            for key in valid_metrics:
                if best_metrics[key]['epoch'] == epoch:
                    sio.savemat(f'{valid_folder}/best_{key}.mat', valid_results)
                    torch.save(save_dict, f'{model_folder}/best_{key}.pth')
                    print(f'\t[Save the best] Model/Data saved to {model_folder}/best_{key}.pth')

                # for key in valid_metrics:
                #     if best_metrics[key]['epoch'] == epoch:
                #         self.config['methods'][method_name]['test']['weight_file'] = f'best_{key}'
                #         self.test(test_dataset=test_dataset, exp_folder=exp_folder, during_train=True)

        #############################################################
        # Stop/Clean
        #############################################################
        print('\nTotal training time: {:.2f} m'.format((time.time() - start_t) / 60))
        writer.close()
        transcript.stop()

    # =================================================================================================================
    # Test
    # =================================================================================================================
    def test(self, test_dataset=None, exp_folder='', during_train=False):
        #############################################################
        # read settings
        #############################################################
        gpu_ids = self.config['settings']['gpu_ids']
        code_root = self.config['settings']['code_root']
        exp_root = self.config['settings']['exp_root']

        fold  = self.config['settings']['fold']
        problem_name = self.config['settings']['problem_name']
        method_name = self.config['settings']['method_name']
        desp = self.config['settings']['description']           

        # method config
        method_config = self.config['methods'][method_name]
        num_workers = method_config['num_workers']

        # test config
        test_config = method_config['test']
        weight_file = test_config['weight_file']
        test_batch_size = test_config['batch_size'] * len(gpu_ids)
        metrics = test_config['metrics']

        #############################################################
        # init settings
        #############################################################
        # define folders
        if not exp_folder: 
            exp_folder = self.config['methods'][method_name]['test']['exp_folder'][f'f{fold}_{test_dataset.noise_type}{test_dataset.noise_level}']
            
        model_folder = f'{exp_folder}/models'

        subj_flag = f'{test_dataset.data_name[0]}'
        motion_flag  = f'{test_dataset.motion_level}'
        fold_flag    = f'f{fold}'
        noise_flag   = f'{test_dataset.noise_type}{test_dataset.noise_level}'
        exp_flag = f'{subj_flag}{motion_flag}_{fold_flag}_{noise_flag}' 
        test_folder = f'{exp_folder}/test/{weight_file}/{exp_flag}'
        
        # make folders
        utils.check_and_mkdir(test_folder)

        # init a logfile
        transcript.start(f'{test_folder}/test_logfile.log', mode='w')

        #############################################################
        # init data
        #############################################################
        # init model data
        if not during_train:
            save_dict = torch.load(f'{model_folder}/{weight_file}.pth')
            self.model = torch.nn.DataParallel(self.model).to(self.device)
            self.model.module.load_state_dict(save_dict['model_state_dict'],strict=True)

        # init test data
        test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, drop_last=False, num_workers=num_workers)

        # init logging data
        print(f'Start testing: model at {model_folder}/{weight_file}.pth')

        #############################################################
        # test
        #############################################################
        test_records = {
            'ipt_metrics': {key: [] for key in metrics},  # record for all images at this epoch
            'pre_metrics': {key: [] for key in metrics},
        }
        test_results = {'pre': [],
                        'ipt_metrics': [],
                        'pre_metrics': []}

        run_time = 0
        self.model.eval()
        with torch.no_grad():
            for counter, batch in enumerate(test_dataloader, 0):
                # get input
                ipt, gdt, *args = (data.to(self.device) for data in batch)

                # perform the forward pass
                start_time = time.time()                
                pre = self.model(ipt, gdt, *args)
                pre_comp = self.DSClass.change_dim(pre, *args, p='to_loss')
                stop_time = time.time()

                # compute time cost
                run_time += stop_time - start_time

                # results
                ipt_metrics = utils.image_metrics(gdt, ipt, metrics=metrics, dataset=self.dataset_name,  mode='multi')
                pre_metrics = utils.image_metrics(gdt, pre_comp, metrics=metrics, dataset=self.dataset_name, mode='multi')

                # update records
                for key in metrics:
                    test_records['ipt_metrics'][key].append(ipt_metrics[key])
                    test_records['pre_metrics'][key].append(pre_metrics[key])

                # update data
                test_results['pre'].append(self.DSClass.change_dim(pre, *args, p='to_save'))
                test_results['ipt_metrics'].append(ipt_metrics)
                test_results['pre_metrics'].append(pre_metrics)

                # print statistic
                print(f'[Test](Batch: [{counter:02d}] / [{len(test_dataloader)}])',
                      f'|| ipt: {str({key: np.round(ipt_metrics[key], 2) for key in metrics})}',
                      f'|| pre: {str({key: np.round(pre_metrics[key], 2) for key in metrics})}')

            # reform the resulst
            test_results['pre'] = np.concatenate(test_results['pre'], axis=0)

            # compute the average
            ipt_metrics_avg = {key: np.mean(test_records['ipt_metrics'][key]) for key in metrics}
            pre_metrics_avg = {key: np.mean(test_records['pre_metrics'][key]) for key in metrics}

            # make a table
            table_title = [list(test_results['ipt_metrics'][0].keys()) + list(test_results['pre_metrics'][0].keys())]
            table_body = [np.round(list(test_results['ipt_metrics'][idx].values()) + list(test_results['pre_metrics'][idx].values()), 2) for idx in range(len(test_results['ipt_metrics']))]
            table_avg = [np.round(list(ipt_metrics_avg.values()) + list(pre_metrics_avg.values()), 2)]
            table = table_title + table_body + table_avg

            # print statistic
            print(f'\t[Test]:',
                  f'|| ipt(avg): {str({key: np.round(ipt_metrics_avg[key], 2) for key in metrics})}',
                  f'|| pre(avg): {str({key: np.round(pre_metrics_avg[key], 2) for key in metrics})}',
                  f'|| time(all): {run_time:.2f} seconds')

            # print table
            print(tabulate(table, headers='firstrow', tablefmt='fancy_grid', showindex=True))

            #############################################################
            # save data
            #############################################################
            sio.savemat(f'{test_folder}/data.mat', test_results)
            source = h5py.File(f'{test_folder}/data.h5', 'w')             
            source.create_dataset(name='pre', data=test_results['pre'])

            # write table
            with open(f'{test_folder}/table.csv', 'w') as csvfile:
                writer = csv.writer(csvfile)
                [writer.writerow(r) for r in table]

            #############################################################
            # stop
            #############################################################
            transcript.stop()
    # =================================================================================================================
    # PNP
    # =================================================================================================================
    def pnp(self, test_dataset=None, exp_folder=''):
        #############################################################
        # read settings
        #############################################################
        gpu_ids = self.config['settings']['gpu_ids']
        code_root = self.config['settings']['code_root']
        exp_root = self.config['settings']['exp_root']

        fold  =  self.config['settings']['fold']
        problem_name = self.config['settings']['problem_name']
        method_name = self.config['settings']['method_name']
        desp = self.config['settings']['description']           

        # method config
        method_config = self.config['methods'][method_name]
        num_workers = method_config['num_workers']
        
        test_batch_size = method_config['batch_size'] * len(gpu_ids)
        metrics = method_config['metrics']

        # dataset
        slice_range = self.config['problems'][problem_name]['test']['slice_range']
        slice_select = self.config['problems'][problem_name]['test']['slice_select']
        slice_to_run = slice_select[str(len(test_dataset))]
        
        #############################################################
        # init settings
        #############################################################
        # define folders
        if not exp_folder:
            A_flag = f'f{fold}'
            dataset_flag = test_dataset.flag()
            exp_flag = f'{A_flag}_{dataset_flag}{desp}'
            reg_flag = f'/{self.model.flag()}'
            exp_folder = f'{exp_root}/{method_name}/{exp_flag}/{reg_flag}'
            
        subj_name = f'{test_dataset.data_name[0]}'
        motion_flag  = f'{test_dataset.motion_level}'
        fold_flag    = f'f{fold}'
        noise_flag   = f'{test_dataset.noise_type}{test_dataset.noise_level}'
        subj_flag = f'{subj_name}{motion_flag}_{fold_flag}_{noise_flag}'

        test_folder = f'{exp_folder}/{subj_flag}'
        utils.check_and_mkdir(test_folder)
        #############################################################
        # test
        #############################################################
        self.model.eval()
        with torch.no_grad():
            for counter in range(slice_to_run[0], slice_to_run[1]): 
                slice_folder = f'{test_folder}/slice_{counter}'
                utils.check_and_mkdir(slice_folder)
                transcript.start(f'{slice_folder}/test_logfile.log', mode='w')

                batch = test_dataset[counter]
                ipt, gdt, *args = (data.unsqueeze(0).to(self.device) for data in batch)
                pre = self.model(ipt, gdt, *args, save_folder=slice_folder)
                
                transcript.stop()

        