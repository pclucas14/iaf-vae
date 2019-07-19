import os
import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torch.nn.utils.weight_norm as wn
from collections import OrderedDict as OD
from collections import defaultdict as DD

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def print_and_save_args(args, path):
    print(args)
    # let's save the args as json to enable easy loading
    with open(os.path.join(path, 'args.json'), 'w') as f: 
        json.dump(vars(args), f)

def load_model_from_file(path):
    with open(os.path.join(path, 'args.json'), 'rb') as f:
        args = dotdict(json.load(f))

    # create model
    from main import VAE
    model = VAE(args)

    # load weights
    model.load_state_dict(torch.load(os.path.join(path, 'best_model.pth')))

    return model

def set_seed(seed):
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def print_and_log_scalar(writer, name, value, write_no, end_token=''):
    if isinstance(value, list):
        if len(value) == 0: return
        
        str_tp = str(type(value[0]))
        if type(value[0]) == torch.Tensor:
            value = torch.mean(torch.stack(value))
        elif 'float' in str_tp or 'int' in str_tp:
            value = sum(value) / len(value)
    zeros = 40 - len(name) 
    name += ' ' * zeros
    print('{} @ write {} = {:.4f}{}'.format(name, write_no, value, end_token))
    writer.add_scalar(name, value, write_no)

def maybe_create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# create logging containers
def reset_log():
    # TODO: make it such that if key is not in dict, it's init with incoming content
    # this way, we won't have to know in advance what we want to monitor
    return DD(list)
    logs = OD()
    for name in ['inner log p(x|z)', 'log p(x|z)', 'log p(x|z) nn', 'commit', 'vq', 'kl', 'bpd', 'elbo']:
        logs[name] = []
    return logs


# loss functions 
# ---------------------------------------------------------------------------------
def logistic_ll(mean, logscale, sample, binsize=1 / 256.0):
    # actually discretized logistic, but who cares
    scale = torch.exp(logscale)
    sample = (torch.floor(sample / binsize) * binsize - mean) / scale
    logp = torch.log(torch.sigmoid(sample + binsize / scale) - torch.sigmoid(sample) + 1e-7)
    return logp.sum(dim=(1,2,3))

def gaussian_ll(mean, logscale, sample):
    logscale = logscale.expand_as(mean)
    dist = D.Normal(mean, torch.exp(logscale))
    logp = dist.log_prob(sample)
    return logp.sum(dim=(1,2,3))
