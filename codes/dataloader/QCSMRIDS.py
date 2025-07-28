from re import T
from this import d
from torch.utils.data import Dataset, DataLoader, Subset
import tqdm
from tools import data_process, utils
import numpy as np
import scipy
from tools import utils
import scipy.io as sio
import torch
from glob import glob
from tqdm import tqdm
import os
import h5py

from DataFidelities.CSMRIDF import CSMRIDF
from DataFidelities.LeBIODF import LeBIODF


class QCSMRIDS(Dataset):
    def __init__(self, msk, config, device, mode='train'):
        super(QCSMRIDS, self).__init__()
        self.msk = msk
        self.config = config
        self.mode = mode
        self.device = device
        self.name = 'qcsmri'

        self.data_root  = self.config['settings']['data_root']
        # dataset config
        dataset_config  = self.config['problems'][self.name]
        self.noise_type = dataset_config['noise_type']
        self.noise_level= dataset_config['noise_level']
        self.bp_method  = dataset_config['bp_method']
        self.scale = dataset_config['scale']

        # mode config
        mode_config       = dataset_config[mode]
        self.data_name    = mode_config['data_name']
        self.slice_range  = mode_config['slice_range']
        self.opt_data     = mode_config['opt_data']
        self.motion_level = mode_config['motion_level']
        self.motion_level = mode_config['motion_level']
        self.is_save      = mode_config['is_save']
        self.sample_index = mode_config['sample_index']

        # define folders
        self.input_rootpath = f'{self.data_root}/GEPCI'
        self.output_rootpath = f'{self.data_root}/GEPCI'

        self.gdts, self.mcks, self.csms, self.bmsks, self.fts, self.index_map = [], [], [], [], [], []
        # generate data
        for subj_index, subj_name in enumerate(self.data_name):
            gdt_file = glob(f'{self.input_rootpath}/{subj_name}/csm_comb.h5')[0]
            gdt_data = data_process.read_h5_files(gdt_file, file_type='csm_comb', opt_data=self.opt_data, slice_range=self.slice_range)
            
            mck_file = glob(f'{self.input_rootpath}/{subj_name}/mck{self.motion_level}.h5')[0]
            mck_data = data_process.read_h5_files(mck_file, file_type=f'mck', opt_data=self.opt_data, slice_range=self.slice_range)

            csm_file = glob(f'{self.input_rootpath}/{subj_name}/csm.h5')[0]
            csm_data = data_process.read_h5_files(csm_file, file_type='csm', opt_data=self.opt_data, slice_range=self.slice_range)

            bmsk_file = glob(f'{self.input_rootpath}/{subj_name}/bmsk.h5')[0]
            bmsk_data = data_process.read_h5_files(bmsk_file, file_type='bmsk', opt_data=self.opt_data, slice_range=self.slice_range)
            
            ft_file = glob(f'{self.input_rootpath}/{subj_name}/F_norm.h5')[0]
            ft_data = data_process.read_h5_files(ft_file, file_type='F_norm', opt_data=self.opt_data, slice_range=self.slice_range)
            
            self.gdts.append(gdt_data)
            self.mcks.append(mck_data)
            self.csms.append(csm_data)
            self.bmsks.append(bmsk_data)
            self.fts.append(ft_data)
            
        # generate indexmap
        if self.opt_data == 'source':
            for subj_index, subj_data in enumerate(self.gdts):
                nslice = len(subj_data)
                slice_selected = self.slice_range[str(nslice)]  if self.slice_range else [0, nslice]
                for slice_index in range(slice_selected[0], slice_selected[1]):
                    self.index_map.append([subj_index, slice_index])
        elif self.opt_data == 'data':
            for subj_index, subj_data in enumerate(self.gdts):
                nslice, necho, ncoil, nheight, nwidth = subj_data.shape
                for slice_index in range(nslice):
                    self.index_map.append([subj_index, slice_index])

        # logging
        print(f'{(self.name)}-{self.mode}: flag     = {self.flag()}')
        print(f'{(self.name)}-{self.mode}: subjects = {len(self.data_name)}, images = {len(self)}')
        print(f'{(self.name)}-{self.mode}: sigSize  : ipt={self.get_sigSize("ipt")}, gdt={self.get_sigSize("gdt")}')
        print(f'{(self.name)}-{self.mode}: argSize  : mea={self.get_sigSize("mea")}, csm={self.get_sigSize("csm")}, ft={self.get_sigSize("ft")}, bmsk={self.get_sigSize("bmsk")}')
        print()

    def __len__(self):
        return len(self.index_map) #count images

    def __getitem__(self, idx):
        subj_index, slice_idx = self.index_map[idx] 

        # e c h w
        csm = torch.from_numpy(self.csms[subj_index][slice_idx]).to(self.device)
        mck = torch.from_numpy(self.mcks[subj_index][slice_idx]).to(self.device)
        # e h w
        gdt = torch.from_numpy(self.gdts[subj_index][slice_idx].squeeze(1)).to(self.device)
        ft = torch.from_numpy(self.fts[subj_index][slice_idx].squeeze(1)).to(self.device)
        bmsk = torch.from_numpy(self.bmsks[subj_index][slice_idx].squeeze(1)).to(self.device)
        # compute MEA/IPT: 
        mea = utils.addwgn_torch(mck, noise_type=self.noise_type, noise_level=self.noise_level)
        mea = self.msk.to(self.device) * mea
        ipt = CSMRIDF.ftran(mea, self.msk, csm, hws_axes=(-2,-1, None), c_axes=-3)

        # scale
        if self.scale:
            scale = torch.mean(torch.abs(gdt[0]))
            ipt = ipt/scale
            gdt = gdt/scale
            mea = mea/scale
        else: 
            scale = torch.tensor(1.0)

        # chaneg dimension: 
        # e, c, h, w --> e, c, h, w, m2
        mea = torch.view_as_real(mea)
        csm = torch.view_as_real(csm)
        # e, h, w --> e, h, w, m2
        ipt = torch.view_as_real(ipt) 
        gdt_no_bmsk = torch.view_as_real(gdt)
        gdt = torch.view_as_real(gdt * bmsk)
        # e, h, w
        ft = torch.abs(ft * bmsk)
        bmsk = torch.abs(bmsk)

        return ipt.float(), gdt.float(), scale.float(), mea.float(), csm.float(), ft.float(), bmsk.float(), gdt_no_bmsk.float()

    def flag(self):
        return f'{self.name}_{self.noise_type}{self.noise_level}'

    def get_sigSize(self, type): # input shape
        data_dict = {'ipt': 0, 'gdt': 1, 'scale':2, 'mea': 3, 'csm': 4, 'ft':5, 'bmsk':6}
        return self.__getitem__(0)[data_dict[type]].shape

    @staticmethod
    def change_dim(data, *args, p='tb_grid', axes=0): # n, e, h, w
        if p == 'tb_grid':
            def normalization(data):
                if len(data.shape) == 5: # N, E, H, W, M
                    data = torch.abs(torch.view_as_complex(data))
                data = torch.abs(data)
                N, E, H, W = data.shape
                data = data.view(-1, 1, H , W)
                data = data - torch.min(data)
                data = data / torch.max(data)
                return data
            if type(data) == tuple: # [mGRE, S0R2s]
                bmsk = args[4]
                S0 = normalization(data[0][:, 0, ...].unsqueeze(1))  * bmsk # N E H W
                R2s = normalization(data[0][:, 1, ...].unsqueeze(1)) * bmsk 
                mGRE = normalization(data[1])   # N E H W M 
                data = torch.cat((mGRE, S0, R2s), 0)
            else:
                data = normalization(data)   
        elif p == 'to_2dcnn': # n, e, h, w, 2 -> n*e, 2, h, w
            N, E, H, W, M = data.shape
            data = data.permute(0, 1, 4, 2, 3)
            data = data.view(-1, M, H, W)
        elif p == 'from_2dcnn': # e * n, 2, h, w ->  n, e h w 2
            EN, M, H, W = data.shape
            E = 10; N = EN//E
            data = data.view(N, E, M, H, W)
            data = data.permute((0, 1, 3, 4, 2))     
        elif p == 'to_loss': # n e10 h w
            ft = args[3]
            S0 =  data[0][:, 0, :, :].unsqueeze(1)
            R2s = data[0][:, 1, :, :].unsqueeze(1)
            data = LeBIODF.fmult(S0, R2s, ft)
        elif p == 'to_save':
            scale = args[0]
            bmsk = args[4]
            if type(data) == tuple: # [mGRE, S0R2s]
                # s e1  h w (real)
                S0R2s = data[0] * bmsk
                S0R2s[:, 0, ...] = S0R2s[:, 0, ...] * scale.view(-1, 1, 1) 
                # s e10 h w (complex)
                mGRE = torch.view_as_complex(data[1]) * scale.view(-1, 1, 1, 1)
                data = torch.cat((S0R2s, mGRE), 1).cpu().numpy()
        return data

    @staticmethod
    def norm_data(data, method='constant', val=None):
        if method == 'none':
            return data, None
        N, E, H, W, M = data.shape
        data_abs = torch.abs(torch.view_as_complex(data))
        if method == 'constant':
            val = torch.mean(data_abs[:,0, :, :].unsqueeze(1), dim=(1, 2, 3), keepdim=True) if val == None else 1 
            data = data / val
        return data, val

    @staticmethod
    def denorm_data(data, method='constant', val=None):
        if method == 'none':
            return data
        if method == 'constant':
            data = data * val
        return data



