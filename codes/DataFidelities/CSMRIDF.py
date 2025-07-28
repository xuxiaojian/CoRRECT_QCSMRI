from email import utils
import torch
import torch.nn.functional as F
import torch.nn as nn
from tools import data_process

class CSMRIDF(nn.Module):

    def __init__(self, msk):
        super(CSMRIDF, self).__init__()
        self.msk = msk

    def eval(self, x, y, csm):
        with torch.no_grad():
            Ax = self.fmult(x, self.msk, csm)
            bs  = y.shape[0]
            d = 1 / (2 * bs) * torch.sum(torch.square(Ax - y))
            return d

    def grad(self, x, y, csm):
        with torch.no_grad():
            Ax = self.fmult(x, self.msk, csm)
            grad_d = self.ftran(Ax - y, self.msk, csm)
            return grad_d

    def name(self):
        return 'csmri'

    @staticmethod # Ax -- > y
    def fmult(x, msk, csm, hws_axes=(-2,-1, None), c_axes=-3): # s, e, c, h, w,
        # FWD_OP = Mask, FFT, Coil (x)
        need_convert = False if torch.is_complex(x) else True
        if need_convert: 
            x = torch.view_as_complex(x)
            csm = torch.view_as_complex(csm)
        Sx = torch.mul(x.unsqueeze(c_axes), csm)
        FSx  = data_process.img_to_ksp(Sx, hws_axes=hws_axes)
        MSx = torch.mul(msk.to(x.device), FSx)
        if need_convert: MSx = torch.view_as_real(MSx)
        return MSx

    @staticmethod # AT(A(x) - y) -- > x
    def ftran(z, msk, csm, hws_axes=(-2,-1, None), c_axes=-3): # s, e, c, h, w
        need_convert = False if torch.is_complex(z) else True
        if need_convert: 
            z = torch.view_as_complex(z)
            csm = torch.view_as_complex(csm)
        # BWD_OP =  I-Coil, IFFT, I-Mask  (y) 
        MTz = torch.mul(z, msk.to(z.device))
        # MTz = z
        FTMTz  = data_process.ksp_to_img(MTz, hws_axes=hws_axes)
        STFTMTz =  torch.sum(torch.mul(FTMTz, torch.conj(csm)), c_axes)
        if need_convert: STFTMTz = torch.view_as_real(STFTMTz)
        return STFTMTz
    
