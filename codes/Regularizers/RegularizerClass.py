from __future__ import print_function, division, absolute_import, unicode_literals

import sys
import os
import shutil
import math
import numpy as np
import logging
import torch

from abc import ABC, abstractmethod
from collections import OrderedDict


############## Basis Class ##############

class RegularizerClass(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def prox(self,z,step,pin):
        pass

    @abstractmethod
    def eval(self,z,step,pin):
        pass

    def name(self):
        pass

