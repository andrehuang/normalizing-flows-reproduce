from __future__ import print_function
import torch
import torch.utils.data
from torch.autograd import Variable

import math

# log N(x| mean, var) = -log sqrt(2pi) -0.5 log var - 0.5 (x-mean)(x-mean)/var

def normal_dist(x, mean, logvar, dim):
    log_norm = -0.5 * (logvar + (x - mean) * (x - mean) * logvar.exp().reciprocal()) 

    return torch.sum(log_norm, dim) ##why??######


