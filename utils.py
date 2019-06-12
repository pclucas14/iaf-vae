import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as wn

def discretized_logistic(mean, logscale, binsize=1 / 256.0, sample=None):
    scale = torch.exp(logscale)
    sample = (torch.floor(sample / binsize) * binsize - mean) / scale
    logp = torch.log(torch.sigmoid(sample + binsize / scale) - torch.sigmoid(sample) + 1e-7)
    return logp.sum(dim=(1,2,3))

def print_and_log_scalar(writer, name, value, write_no, end_token=''):
    if isinstance(value, list):
        if len(value) == 0: return 
        value = torch.mean(torch.stack(value))
    zeros = 40 - len(name) 
    name += ' ' * zeros
    print('{} @ write {} = {:.4f}{}'.format(name, write_no, value, end_token))
    writer.add_scalar(name, value, write_no)

def maybe_create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
