import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from os.path import join
from tensorboardX import SummaryWriter
from torchvision.utils import save_image
from collections import OrderedDict as OD 
from torchvision import datasets, transforms, utils

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
    
    
    def cond_sample(self, input):
        # assumes input is \in [-0.5, 0.5] 
        x = self.first_conv(input)
        kl, kl_obj = 0., 0.

        h = self.h.view(1, -1, 1, 1)

        for layer in self.layers:
            for sub_layer in layer:
                x = sub_layer.up(x)

        h = h.expand_as(x)
        self.hid_shape = x[0].size()

        outs = []

        current = 0
        for i, layer in enumerate(reversed(self.layers)):
            for j, sub_layer in enumerate(reversed(layer)):
                h, curr_kl, curr_kl_obj = sub_layer.down(h)
                
                h_copy = h
                again = 0
                # now, sample the rest of the way:
                for layer_ in reversed(self.layers):
                    for sub_layer_ in reversed(layer_):
                        if again > current:
                            h_copy, _, _ = sub_layer_.down(h_copy, sample=True)
                        
                        again += 1
                        
                x = F.elu(h_copy)
                x = self.last_conv(x)
                x = x.clamp(min=-0.5 + 1. / 512., max=0.5 - 1. / 512.)
                outs += [x]

                current += 1

        return outs
        
# Main
# ----------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--n_blocks', type=int, default=4)
    parser.add_argument('--depth', type=int, default=2)
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

    # reproducibility is da best
    set_seed(0)

    opt = torch.optim.Adamax(model.parameters(), lr=args.lr)

    # create datasets / dataloaders
    scale_inv = lambda x : x + 0.5
    ds_transforms = transforms.Compose([transforms.ToTensor(), lambda x : x - 0.5])
    kwargs = {'num_workers':1, 'pin_memory':True, 'drop_last':True}

    train_loader = torch.utils.data.DataLoader(datasets.CIFAR10('../cl-pytorch/data', train=True, 
        download=True, transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader  = torch.utils.data.DataLoader(datasets.CIFAR10('../cl-pytorch/data', train=False, 
        download=True, transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)

    # spawn writer
    model_name = 'NB{}_D{}_Z{}_H{}_BS{}_FB{}_LR{}_IAF{}'.format(args.n_blocks, args.depth, args.z_size, args.h_size, 
                                                                args.batch_size, args.free_bits, args.lr, args.iaf)

    model_name = 'test' if args.debug else model_name
    log_dir    = join('runs', model_name)
    sample_dir = join(log_dir, 'samples')
    writer     = SummaryWriter(log_dir=log_dir)
    maybe_create_dir(sample_dir)

    print_and_save_args(args, log_dir)
    print('logging into %s' % log_dir)
    maybe_create_dir(sample_dir)
    best_test = float('inf')


    print('starting training')
    for epoch in range(args.n_epochs):
        model.train()
        train_log = reset_log()

        for batch_idx, (input,_) in enumerate(train_loader):

            input = input.cuda()
            x, kl, kl_obj = model(input)

            log_pxz = logistic_ll(x, model.dec_log_stdv, sample=input)
            loss = (kl_obj - log_pxz).sum() / x.size(0)
            elbo = (kl     - log_pxz)
            bpd  = elbo / (32 * 32 * 3 * np.log(2.))
         
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_log['kl']         += [kl.mean()]
            train_log['bpd']        += [bpd.mean()]
            train_log['elbo']       += [elbo.mean()]
            train_log['kl obj']     += [kl_obj.mean()]
            train_log['log p(x|z)'] += [log_pxz.mean()]

        for key, value in train_log.items():
            print_and_log_scalar(writer, 'train/%s' % key, value, epoch)
        print()
        
        model.eval()
        test_log = reset_log()

        with torch.no_grad():
            for batch_idx, (input,_) in enumerate(test_loader):
                input = input.cuda()
                x, kl, kl_obj = model(input)
            
                log_pxz = logistic_ll(x, model.dec_log_stdv, sample=input)
                loss = (kl_obj - log_pxz).sum() / x.size(0)
                elbo = (kl     - log_pxz)
                bpd  = elbo / (32 * 32 * 3 * np.log(2.))
                
                test_log['kl']         += [kl.mean()]
                test_log['bpd']        += [bpd.mean()]
                test_log['elbo']       += [elbo.mean()]
                test_log['kl obj']     += [kl_obj.mean()]
                test_log['log p(x|z)'] += [log_pxz.mean()]
                
            all_samples = model.cond_sample(input)
            # save reconstructions
            out = torch.stack((x, input))               # 2, bs, 3, 32, 32
            out = out.transpose(1,0).contiguous()       # bs, 2, 3, 32, 32
            out = out.view(-1, x.size(-3), x.size(-2), x.size(-1))
           
            all_samples += [x]
            all_samples = torch.stack(all_samples)     # L, bs, 3, 32, 32
            all_samples = all_samples.transpose(1,0)
            all_samples = all_samples.contiguous()     # bs, L, 3, 32, 32
            all_samples = all_samples.view(-1, x.size(-3), x.size(-2), x.size(-1))

            save_image(scale_inv(all_samples), join(sample_dir, 'test_levels_{}.png'.format(epoch)), nrow=12)
            save_image(scale_inv(out), join(sample_dir, 'test_recon_{}.png'.format(epoch)), nrow=12)
            save_image(scale_inv(model.sample(64)), join(sample_dir, 'sample_{}.png'.format(epoch)), nrow=8)
            

        for key, value in test_log.items():
            print_and_log_scalar(writer, 'test/%s' % key, value, epoch)
        print()
        
        current_test = sum(test_log['bpd']) / batch_idx
        if current_test < best_test:
            best_test = current_test
            print('saving best model')
            torch.save(model.state_dict(), join(log_dir, 'best_model.pth'))
