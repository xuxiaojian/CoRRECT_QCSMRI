from email import utils
import torch
import torch.nn.functional as F
import torch.nn as nn
from tools import data_process
import numpy as np

class LeBIODF(nn.Module):
    def __init__(self, te):
        super(LeBIODF, self).__init__()
        self.te = te

    def eval(self, x, y):
        pass

    def grad(self, x, y):
        pass

    def get_y(self, x, ft):
        self.fmult(x, self.te, ft)

    def name(self):
        return 'lebio'

    @staticmethod # Ax -- > y
    def fmult(S0, R2s, ft, te=None): # s, e, h, w,
        if te is None:  
            te = torch.from_numpy(np.array(range(4, 44, 4)) / 1e3).float()
        TE = te[None, :, None, None]
        mGRE = S0 * torch.exp(-R2s * TE.to(S0.device))  * ft
        return mGRE

    @staticmethod 
    def ftran(z, te, ft): 
        pass
    
