import time
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, utils
from torchvision.utils import save_image

from layers import IAFLayer
from utils  import * 

# Model definition
# ----------------------------------------------------------------------------------------------
class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        self.register_parameter('h', torch.nn.Parameter(torch.zeros(args.h_size)))
        self.register_parameter('dec_log_stdv', torch.nn.Parameter(torch.Tensor([0.])))

        layers = []
        # build network
        for i in range(args.depth):
            layer = []

            for j in range(args.n_blocks):
                downsample = (i > 0) and (j == 0)
                layer += [IAFLayer(args, downsample)]

            layers += [nn.ModuleList(layer)]

        self.layers = nn.ModuleList(layers) 
        
        self.first_conv = nn.Conv2d(3, args.h_size, 4, 2, 1)
        self.last_conv = nn.ConvTranspose2d(args.h_size, 3, 4, 2, 1)

    def forward(self, input):
        # assumes input is \in [-0.5, 0.5] 
        x = self.first_conv(input)
        kl, kl_obj = 0., 0.

        h = self.h.view(1, -1, 1, 1)

        for layer in self.layers:
            for sub_layer in layer:
                x = sub_layer.up(x)

        h = h.expand_as(x)
        self.hid_shape = x[0].size()

        for layer in reversed(self.layers):
            for sub_layer in reversed(layer):
                h, curr_kl, curr_kl_obj = sub_layer.down(h)
                kl     += curr_kl
                kl_obj += curr_kl_obj

        x = F.elu(h)
        x = self.last_conv(x)
        
        x = x.clamp(min=-0.5 + 1. / 512., max=0.5 - 1. / 512.)

        return x, kl, kl_obj


    def sample(self, n_samples=64):
        h = self.h.view(1, -1, 1, 1)
        h = h.expand((n_samples, *self.hid_shape))
        
        for layer in reversed(self.layers):
            for sub_layer in reversed(layer):
                h, _, _ = sub_layer.down(h, sample=True)

        x = F.elu(h)
        x = self.last_conv(x)
        
        return x.clamp(min=-0.5 + 1. / 512., max=0.5 - 1. / 512.)
        
# Main
# ----------------------------------------------------------------------------------------------

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_blocks', type=int, default=20)
parser.add_argument('--depth', type=int, default=1)
parser.add_argument('--z_size', type=int, default=32)
parser.add_argument('--h_size', type=int, default=64)
parser.add_argument('--n_epochs', type=int, default=1000)
parser.add_argument('--warmup', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--free_bits', type=float, default=0.1)
parser.add_argument('--iaf', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-3)
args = parser.parse_args()

# create model and ship to GPU
model = VAE(args).cuda()
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# create datasets / dataloaders
scale_inv = lambda x : x + 0.5
ds_transforms = transforms.Compose([transforms.ToTensor(), lambda x : x - 0.5])
kwargs = {'num_workers':1, 'pin_memory':True, 'drop_last':True}

train_loader = torch.utils.data.DataLoader(datasets.CIFAR10('../cl-pytorch/data', train=True, 
    download=True, transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader  = torch.utils.data.DataLoader(datasets.CIFAR10('../cl-pytorch/data', train=False, 
                transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)

print('starting training')
for epoch in range(args.n_epochs):
    model.train()
    train_loss, train_re, train_kl = 0., 0., 0.
    time_ = time.time()

    for batch_idx, (input,_) in enumerate(train_loader):
       
        input = input.cuda()
        x, kl, kl_obj = model(input)

        log_pxz = discretized_logistic(x, model.dec_log_stdv, sample=input)
        loss = (kl_obj - log_pxz).sum()
        elbo = (kl     - log_pxz).sum()
        
        optimizer.zero_grad()
        (loss / input.size(0)).backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_re   += -log_pxz.sum().item()
        train_kl   += kl.sum().item()

        if batch_idx % 50 == 49 : 
            deno = 50 * args.batch_size * 32 * 32 * 3 * np.log(2.)
            print('train loss : {:.4f}\t recon: {:.4f}\t kl : {:.4f}\t elbo : {:.4f}\t time : {:.4f}'.format(
                (train_loss / (50 * args.batch_size)), 
                (train_re) / (50 * args.batch_size),
                (train_kl / (50 * args.batch_size)),
                (train_re + train_kl) / deno,
                (time.time() - time_)))
           
            # break
            train_loss, train_re, train_kl = 0., 0., 0.
            time_ = time.time()
    
    model.eval()
    test_loss, test_re, test_kl = 0., 0., 0.

    print('test time!')
    time_ = time.time()
    with torch.no_grad():
        for batch_idx, (input,_) in enumerate(test_loader):
            input = input.cuda()
            x, kl, kl_obj = model(input)
        
            log_pxz = discretized_logistic(x, model.dec_log_stdv, sample=input)
            loss = (kl_obj - log_pxz).sum()
            elbo = (kl     - log_pxz).sum()
            
            test_loss += loss.item()
            test_re   += -log_pxz.sum().item()
            test_kl   += kl.sum().item()

        out = torch.stack((x, input)) # 2, bs, 3, 32, 32
        out = out.transpose(1,0).contiguous() # bs, 2, 3, 32, 32
        out = out.view(-1, x.size(-3), x.size(-2), x.size(-1))
        
        save_image(scale_inv(out), 'samples/test_recon_{}.png'.format(epoch), nrow=12)
        save_image(scale_inv(model.sample(64)), 'samples/sample_{}.png'.format(epoch), nrow=8)

        deno = batch_idx * args.batch_size * 32 * 32 * 3 * np.log(2.)
        print('test loss : {:.4f}\t recon: {:.4f}\t kl : {:.4f}\t elbo : {:.4f}\t time : {:.4f}'.format(
            (test_loss / (batch_idx * args.batch_size)), 
            (test_re) / (batch_idx * args.batch_size),
            (test_kl / (batch_idx * args.batch_size)),
            (test_re + test_kl) / deno,
            (time.time() - time_)))
        

