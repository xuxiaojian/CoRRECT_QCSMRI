from tkinter.dnd import DndHandler
from tools.utils import *
import os
import numpy as np
import random
import torch
from DataFidelities.CSMRIDF import CSMRIDF
# import cv2

def get_image_paths(data_path, Nimgs=None):
    IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif']
    if data_path is not None:
        assert os.path.isdir(data_path), '{:s} is not a valid directory'.format(data_path)
        images = []
        for dirpath, _, fnames in sorted(os.walk(data_path)):
            for fname in sorted(fnames):
                if any(fname.endswith(extension) for extension in IMG_EXTENSIONS) and fname[:2] != '._':
                    img_path = os.path.join(dirpath, fname)
                    images.append(img_path)
        assert images, '{:s} has no valid image file'.format(data_path)
    paths = sorted(images)[:Nimgs]
    return paths


def patches_from_image(img, p_size=512, p_overlap=64, mode='multi'):
    w, h = img.shape[:2]
    patches = []

    if p_size and p_size <= w and p_size <= h and p_overlap:
        w1 = list(np.arange(0, w - p_size, p_size - p_overlap, dtype=np.int))
        h1 = list(np.arange(0, h - p_size, p_size - p_overlap, dtype=np.int))
        w1.append(w - p_size)
        h1.append(h - p_size)
        if mode == 'multi':
            for i in w1:
                for j in h1:
                    patches.append(img[i:i+p_size, j:j+p_size,:])
        elif mode == 'single':
            i = random.choice(w1)
            j = random.choice(h1)
            patches.append(img[i:i+p_size, j:j+p_size, :])
    elif p_size and not p_overlap:
        offset = (w - p_size) // 2
        patches.append(img[offset:w - offset, offset:h - offset, :])
    else:
        patches.append(img)

    return patches

def augment_img(img):
    mode = np.random.randint(0, 8)
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))

# convert uint to 3-dimensional torch tensor
def uint2tensor(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.)

##################################################################
# read from .dat files
##################################################################
import mapvbvd
import  scipy.io as sio
from pathlib import Path
from glob import glob
from pathlib import Path
import csv
import h5py


# all output follows : s, e, c, h, w
def read_mri_files(file_path, file_type='ori', slice_range={}, coil_range={}, echo_range={}, opt_type='numpy', data_format='SECHW', flagRemoveOS=False):
    # print('\nLoading {}: {}'.format(file_type, file_path))

    # Check shape for reference info
    if file_type != 'msk':
        folder_path = Path(file_path).parent.absolute()
        shp_file_path = f'{folder_path}/shp.csv'
        with open(shp_file_path, 'r') as csvfile:
                    reader = csv.reader(csvfile)
                    header_line = next(reader)
                    data_line = next(reader)
                    nslice, necho, ncoil, nheight, nwidth = tuple([int(item) for item in data_line])

        coil_selected = coil_range[str(ncoil)]  if coil_range else [0, ncoil]
        echo_selected = echo_range[str(necho)]  if echo_range else [0, necho]
        slice_selected = slice_range[str(nslice)]  if slice_range else [0, nslice]
                
    # Load data
    if file_type == 'ori': # h, c, w, s, e ---> s, e, c, h, w
        twix_obj = mapvbvd.mapVBVD(file_path)
        twix_obj.image.squeeze = True
        twix_obj.image.flagRemoveOS = flagRemoveOS             

        data = twix_obj.image['']
        data = np.transpose(data, (3, 4, 1, 0, 2))
        data = data[slice_selected[0]:slice_selected[1], echo_selected[0]:echo_selected[1], coil_selected[0]:coil_selected[1], :, :]
        
    elif file_type == 'mck': # s, e, c, h, w
        with h5py.File(file_path, 'r', swmr=True) as mckfile:
            data = mckfile['mck'][slice_selected[0]:slice_selected[1], :,:,:, :]

    elif file_type == 'csm': # s, e, c, h, w 
        data = sio.loadmat(file_path, verify_compressed_data_integrity=False)['csm']
        data = data[slice_selected[0]:slice_selected[1], echo_selected[0]:echo_selected[1], :,:, :]
    
    elif file_type == 'bmsk':#  h, w, s --> s, e, c, h, w   
        data = sio.loadmat(file_path, verify_compressed_data_integrity=False)['Mask']
        data = np.transpose(data, (2, 0, 1))[slice_selected[0]:slice_selected[1], None, None,:, :]

    elif file_type == 'csm_comb':# s, e, c, h, w  
        data = sio.loadmat(file_path, verify_compressed_data_integrity=False)['csm_comb']
        data = data[slice_selected[0]:slice_selected[1], echo_selected[0]:echo_selected[1], :,:, :]

    elif file_type == 'ft':#  h, w, s, e --> s e h w --> s, e, c, h, w   
        data = sio.loadmat(file_path, verify_compressed_data_integrity=False)['F_norm']
        data = np.transpose(data, (2, 3, 0, 1))[slice_selected[0]:slice_selected[1], echo_selected[0]:echo_selected[1], None, :, :]

    elif file_type == 'coe': # h, w, s, c  --> s c h w --> s, e, c, h, w
        data = sio.loadmat(file_path, verify_compressed_data_integrity=False)
        csm, scale = data['csm'], data['scale'] 
        csm, scale = np.transpose(csm, (2, 3, 0, 1))[:, None, :, :, :], np.transpose(scale, (2, 3, 0, 1))[:, None, :, :, :]
        csm, scale = csm[slice_selected[0]:slice_selected[1], :, :,:, :] , scale[slice_selected[0]:slice_selected[1], :, :,:, :] 
        data = (csm, scale)
    
    elif file_type == 'ima_comb':#  h, w, s, e --> s e h w --> s, e, c, h, w  
        data = sio.loadmat(file_path, verify_compressed_data_integrity=False)['ima_comb']
        data = np.transpose(data, (2, 3, 0, 1))[slice_selected[0]:slice_selected[1], echo_selected[0]:echo_selected[1], None, :, :]

    elif file_type == 'msk': # h, w
        data = sio.loadmat(file_path, verify_compressed_data_integrity=False)['msk'][0]
        data = data[None, None, None, :, :]

    elif file_type == 'shp':
        with open(file_path, 'r') as csvfile:
                reader = csv.reader(csvfile)
                header_line = next(reader)
                data_line = next(reader)
                data = tuple([int(item) for item in data_line])

    if data_format == 'SCEHW': data = np.transpose(data, (3, 1, 4, 0, 2))
    if opt_type == 'tensor': data = torch.tensor(data)
    if file_type == 'msk': data = data.squeeze(0)
    return data

def read_h5_files(file_path, file_type='csm_comb', slice_range={}, coil_range={}, echo_range={}, opt_data='source', opt_type='numpy', data_format='SECHW'):
    source = h5py.File(file_path, 'r', swmr=True)[file_type]
    if opt_data == 'source': 
        return source
    elif opt_data == 'data':
        nslice = len(source)
        necho, ncoil, nheight, nwidth = source[0].shape
        coil_selected = coil_range[str(ncoil)]  if coil_range else [0, ncoil]
        echo_selected = echo_range[str(necho)]  if echo_range else [0, necho]
        slice_selected = slice_range[str(nslice)]  if slice_range else [0, nslice]
        
        data = source[slice_selected[0]:slice_selected[1], echo_selected[0]:echo_selected[1], coil_selected[0]:coil_selected[1],:, :]
        if data_format == 'SCEHW': data = np.transpose(data, (3, 1, 4, 0, 2))
        if opt_type == 'tensor': data = torch.tensor(data)
        return  data
    else:
        print('Type {opt_type} not found !')
        exit()

##################################################################
# img to ksp
##################################################################
import numpy.fft as nft
import torch.fft as tft

def img_to_ksp(data, hws_axes=(-2, -1, 0)): # ie: h, w, s, e

    h_ax, w_ax, s_ax = hws_axes
    # do it along s, w, h, 
    if torch.is_tensor(data):
        fft_fun = tft
        fft_ds = lambda img, s_ax, nslice:  fft_fun.fftshift(fft_fun.fft(fft_fun.fftshift(img, dim=s_ax), n=nslice, dim=s_ax), dim=s_ax)
        fft_dw = lambda img, w_ax, nwidth:  fft_fun.fftshift(fft_fun.ifft(fft_fun.fftshift(img, dim=w_ax), n=nwidth, dim=w_ax), dim=w_ax)
        fft_dh = lambda img, h_ax, nheight: fft_fun.fftshift(fft_fun.fft(fft_fun.fftshift(img, dim=h_ax), n=nheight, dim=h_ax), dim=h_ax)
    else:
        fft_fun = nft
        fft_ds = lambda img, s_ax, nslice:  fft_fun.fftshift(fft_fun.fft(fft_fun.fftshift(img, axes=s_ax), n=nslice, axis=s_ax), axes=s_ax)
        fft_dw = lambda img, w_ax, nwidth:  fft_fun.fftshift(fft_fun.ifft(fft_fun.fftshift(img, axes=w_ax), n=nwidth, axis=w_ax), axes=w_ax)
        fft_dh = lambda img, h_ax, nheight: fft_fun.fftshift(fft_fun.fft(fft_fun.fftshift(img, axes=h_ax), n=nheight, axis=h_ax), axes=h_ax)

    if s_ax is not None:
        nslice = data.shape[s_ax]
        data = fft_ds(data, s_ax, nslice)
    if w_ax is not None:
        nwidth = data.shape[w_ax]
        data = fft_dw(data, w_ax, nwidth)
    if h_ax is not None:
        nheight = data.shape[h_ax]
        data = fft_dh(data, h_ax, nheight)

    return data

##################################################################
# ksp to img
##################################################################
def ksp_to_img(data, hws_axes=(-2, -1, 0)): # ie: h, w s, e
    h_ax, w_ax, s_ax = hws_axes

    # do it along s, w, h, 
    if torch.is_tensor(data):
        fft_fun = tft 
        fft_ds = lambda ksp, s_ax, nslice:  fft_fun.fftshift(fft_fun.ifft(fft_fun.fftshift(ksp, dim=s_ax), n=nslice, dim=s_ax), dim=s_ax)
        fft_dw = lambda ksp, w_ax, nwidth:  fft_fun.fftshift(fft_fun.fft(fft_fun.fftshift(ksp, dim=w_ax), n=nwidth, dim=w_ax), dim=w_ax)
        fft_dh = lambda ksp, h_ax, nheight: fft_fun.fftshift(fft_fun.ifft(fft_fun.fftshift(ksp, dim=h_ax), n=nheight, dim=h_ax), dim=h_ax)
    else:
        fft_fun = nft 
        fft_ds = lambda ksp, s_ax, nslice:  fft_fun.fftshift(fft_fun.ifft(fft_fun.fftshift(ksp, axes=s_ax), n=nslice, axis=s_ax), axes=s_ax)
        fft_dw = lambda ksp, w_ax, nwidth:  fft_fun.fftshift(fft_fun.fft(fft_fun.fftshift(ksp, axes=w_ax), n=nwidth, axis=w_ax), axes=w_ax)
        fft_dh = lambda ksp, h_ax, nheight: fft_fun.fftshift(fft_fun.ifft(fft_fun.fftshift(ksp, axes=h_ax), n=nheight, axis=h_ax), axes=h_ax)

    if s_ax is not None:
        nslice = data.shape[s_ax]
        data = fft_ds(data, s_ax, nslice)
    if w_ax is not None:
        nwidth = data.shape[w_ax]
        data = fft_dw(data, w_ax, nwidth)
    if h_ax is not None:
        nheight = data.shape[h_ax]
        data = fft_dh(data, h_ax, nheight)

    return data

##################################################################
# add  motion to data
##################################################################
import scipy
import os, time, datetime
def get_rand_int(data_range, size=None):
    rand_num = np.random.random_integers(data_range[0], data_range[1], size=size)
    return rand_num

def add_motion(data, level='rand', mode='constant', domain='img', device=None, live_settings={}): # mck data s, e, c, h, w

    if live_settings:
        nmotion = live_settings['nmotion']
        w_rng = live_settings['w_rng']
        shift_h_rng = live_settings['shift_h_rng']
        shift_w_rng = live_settings['shift_w_rng']
        rotate_rng = live_settings['rotate_rng']
        motion_start_pos = live_settings['motion_start_pos']
    elif level in ['rand', 'low', 'mid', 'high']:
        motion_rng =  {'rand': [1, 10], 'low':[6, 6], 'mid': [8, 8], 'high':[12, 12]}
        nmotion = get_rand_int(motion_rng[level])
        if level == 'rand':
            w_rng = [[1, 10]] *   nmotion         #[1, 5] [1,10][1, 15]
            shift_h_rng = [[-15, 15]]  *  nmotion  #[-5, 5] [-15, 15] [-20, 20]
            shift_w_rng = [[-15, 15]]  *  nmotion  #[-5, 5] [-15, 15] [-20, 20]
            rotate_rng = [[-15, 15]]   *  nmotion  #[-10, 10] [-15, 15][-30, 30]
            motion_start_pos = [np.concatenate((np.arange(0, 192 // 2 - 15), np.arange(192 // 2 + 15, 191)), axis=0)] * nmotion
        elif level in ['low', 'mid', 'high']:
            shift_h_rng = [[5, 5]] * nmotion
            shift_w_rng = [[5, 5]] * nmotion
            rotate_rng = [[5, 5]]  * nmotion
            w_rng = [[1, 1]] * nmotion
            motion_start_pos = [[68, 68],[70, 70],[72, 72],[74, 74],[76, 76], [78, 78],[73, 73],[75, 75],[120, 120],[128, 128],[16, 16],[144, 144]]
            # 'fx6': {'_low': {'nmotion':6, 'w':1, 'shfit': 5, 'rotate': 5, 'motion_start_pos':[[68, 68],[70, 70],[72, 72],[74, 74],[76, 76], [78, 78]]},
            # '_mid': {'nmotion':8, 'w':1, 'shfit': 5, 'rotate': 5, 'motion_start_pos':        [[68, 68],[70, 70],[72, 72],[74, 74],[76, 76], [78, 78],[73, 73],[75, 75],]},
            # '_high':{'nmotion':12, 'w':1, 'shfit': 5, 'rotate': 5, 'motion_start_pos':       [[68, 68],[70, 70],[72, 72],[74, 74],[76, 76], [78, 78],[73, 73],[75, 75],[120, 120],[128, 128],[16, 16],[144, 144]]},
            # '_live':{'nmotion':0, 'w':0, 'shfit': 0, 'rotate': 0, 'motion_start_pos':[]},


    else:
        print(f'Motion level {level}&live_settings not found !')
        exit(0)

    print("Compute the image ....")
    img = ksp_to_img(data, hws_axes=(-2, -1, None))

    print("Compute the motion ....")
    start_t = time.time()

    # GPU 
    nslice, necho, ncoil, nheight, nwidth = img.shape
    motion_msk = torch.zeros((nheight, nwidth))
    for motion in range(nmotion): # mck data s, e, c, h, w
        # random_num = random.choice(test_list)
        # shift = np.random.choice(test_list)

        shift = (get_rand_int(shift_h_rng[motion]), get_rand_int(shift_w_rng[motion]))
        rotate = get_rand_int(rotate_rng[motion])
        w_start = motion_start_pos[motion][get_rand_int([0, len(motion_start_pos[motion]) - 1])]
        w_end = w_start + get_rand_int(w_rng[motion])
        motion_msk[:,w_start:w_end] = 1
        for s in range(nslice): 
            print(f"****Adding motion with motion={motion}/{nmotion}: shift={shift}, rotate={rotate}, range={w_start}-{w_end}")
            grid_slice = generate_affine_grid((necho, ncoil, nheight, nwidth), translation=shift, rotate=rotate).double().to(device)
            img_slice = torch.from_numpy(img[s, ...]).to(device)
            img_tmp_real = f.grid_sample(img_slice.real, grid_slice)
            img_tmp_imag = f.grid_sample(img_slice.imag, grid_slice)
            img_tmp = torch.complex(img_tmp_real, img_tmp_imag).unsqueeze(0).cpu().numpy()
            ksp_tmp = img_to_ksp(img_tmp, hws_axes=(-2, -1, None))
            data[:, :, :, :, w_start:w_end] = ksp_tmp[:, :, :, :, w_start:w_end]

    print('Total time: {:.2f} m \n'.format((time.time() - start_t) / 60))
    return data, motion_msk

from torch.nn import functional as f
def generate_affine_grid(imgSize, translation=(0, 0), reflection=(1, 1), scale=1, rotate=0, shear=(0, 0)):
    T_translation = np.array([
        [1, 0, translation[0]],
        [0, 1, translation[1]],
        [0, 0, 1]
    ]).astype(np.float32)

    T_reflection = np.array([
        [reflection[0], 0, 0],
        [0, reflection[1], 0],
        [0, 0, 1]
    ]).astype(np.float32)

    T_scale = np.array([
        [scale, 0, 0],
        [0, scale, 0],
        [0, 0, 1]
    ]).astype(np.float32)

    rotate = rotate / 180 * np.pi
    T_rotate = np.array([
        [np.cos(rotate), -np.sin(rotate), 0],
        [np.sin(rotate), np.cos(rotate), 0],
        [0, 0, 1]
    ]).astype(np.float32)

    T_shear = np.array([
        [1, shear[0], 0],
        [shear[1], 1, 0],
        [0, 0, 1]
    ]).astype(np.float32)

    rec = np.matmul(np.matmul(np.matmul(np.matmul(T_translation, T_reflection), T_scale), T_rotate), T_shear)
    rec = rec[:2, :]
    rec = torch.from_numpy(rec)
    theta = rec.unsqueeze_(0)

    return f.affine_grid(theta=theta.repeat(imgSize[0], 1, 1), size=imgSize)

def shiftnrotate(data, shift, rotate, domain='img', mode='constant'):
        if domain == 'ksp':
            angle = np.angle(data); 
            mag = np.abs(data)
            # shift
            data  =  mag * np.exp(1j * (angle + shift/180 * np.pi))
            # rotate
            real = data.real
            imag = data.imag
            data_real = scipy.ndimage.rotate(real, rotate, reshape=False, mode=mode, cval=0.0)
            data_imag = scipy.ndimage.rotate(imag, rotate, reshape=False, mode=mode, cval=0.0)
            data = np.complex(data_real, data_imag)
        elif domain == 'img': 
            data_real = data.real
            data_imag = data.imag
            # shift
            if not all(v == 0 for v in shift):
                data_real = scipy.ndimage.shift(data_real, shift, mode=mode, cval=0.0)
                data_imag = scipy.ndimage.shift(data_imag, shift, mode=mode, cval=0.0)
            # rotate
            if rotate != 0:
                data_real = scipy.ndimage.rotate(data_real, rotate, reshape=False, mode=mode, cval=0.0)
                data_imag = scipy.ndimage.rotate(data_imag, rotate, reshape=False, mode=mode, cval=0.0)

        data = data_real + 1j * data_imag
        return data


    

