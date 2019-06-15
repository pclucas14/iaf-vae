import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as wn
import torch.distributions as D

# Basic Layers 
# -------------------------------------------------------------------------------------------------------

# taken from https://github.com/jzbontar/pixelcnn-pytorch
class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)

class ARMultiConv2d(nn.Module):
    def __init__(self, n_h, n_out, args, nl=F.elu):
        super(ARMultiConv2d, self).__init__()
        self.nl = nl

        convs, out_convs = [], []

        for i, size in enumerate(n_h):
            convs     += [MaskedConv2d('A' if i == 0 else 'B', args.z_size if i == 0 else args.h_size, args.h_size, 3, 1, 1)]
        for i, size in enumerate(n_out):
            out_convs += [MaskedConv2d('B', args.h_size, args.z_size, 3, 1, 1)]

        self.convs = nn.ModuleList(convs)
        self.out_convs = nn.ModuleList(out_convs)


    def forward(self, x, context):
        for i, conv_layer in enumerate(self.convs):
            x = conv_layer(x)
            if i == 0: 
                x += context
            x = self.nl(x)

        return [conv_layer(x) for conv_layer in self.out_convs]


# IAF building block
# -------------------------------------------------------------------------------------------------------

class IAFLayer(nn.Module):
    def __init__(self, args, downsample):
        super(IAFLayer, self).__init__()
        n_in  = args.h_size
        n_out = args.h_size * 2 + args.z_size * 2
        
        self.z_size = args.z_size
        self.h_size = args.h_size
        self.iaf    = args.iaf
        self.ds     = downsample
        self.args   = args

        if downsample:
            stride, padding, filter_size = 2, 1, 4
            self.down_conv_b = wn(nn.ConvTranspose2d(args.h_size + args.z_size, args.h_size, 4, 2, 1))
        else:
            stride, padding, filter_size = 1, 1, 3
            self.down_conv_b = wn(nn.Conv2d(args.h_size + args.z_size, args.h_size, 3, 1, 1))

        # create modules for UP pass: 
        self.up_conv_a = wn(nn.Conv2d(n_in, n_out, filter_size, stride, padding))
        self.up_conv_b = wn(nn.Conv2d(args.h_size, args.h_size, 3, 1, 1))

        # create modules for DOWN pass: 
        self.down_conv_a  = wn(nn.Conv2d(n_in, 4 * self.z_size + 2 * self.h_size, 3, 1, 1))

        if args.iaf:
            self.down_ar_conv = ARMultiConv2d([args.h_size] * 2, [args.z_size] * 2, args)


    def up(self, input):
        x = F.elu(input)
        out_conv = self.up_conv_a(x)
        self.qz_mean, self.qz_logsd, self.up_context, h = out_conv.split([self.z_size] * 2 + [self.h_size] * 2, 1)

        h = F.elu(h)
        h = self.up_conv_b(h)

        if self.ds:
            input = F.upsample(input, scale_factor=0.5)

        return input + 0.1 * h
        

    def down(self, input, sample=False):
        x = F.elu(input)
        x = self.down_conv_a(x)
        
        pz_mean, pz_logsd, rz_mean, rz_logsd, down_context, h_det = x.split([self.z_size] * 4 + [self.h_size] * 2, 1)
        prior = D.Normal(pz_mean, torch.exp(2 * pz_logsd))
            
        if sample:
            z = prior.rsample()
            kl = kl_obj = torch.zeros(input.size(0)).to(input.device)
        else:
            posterior = D.Normal(rz_mean + self.qz_mean, torch.exp(rz_logsd + self.qz_logsd))
            
            z = posterior.rsample()
            logqs = posterior.log_prob(z) 
            context = self.up_context + down_context

            if self.iaf:
                x = self.down_ar_conv(z, context) 
                arw_mean, arw_logsd = x[0] * 0.1, x[1] * 0.1
                z = (z - arw_mean) / torch.exp(arw_logsd)
            
                # the density at the new point is the old one + determinant of transformation
                logq = logqs
                logqs += arw_logsd

            logps = prior.log_prob(z) 
            kl = logqs - logps

            # free bits (doing as in the original repo, even if weird)
            kl_obj = kl.sum(dim=(-2, -1)).mean(dim=0, keepdim=True)
            kl_obj = kl_obj.clamp(min=self.args.free_bits)
            kl_obj = kl_obj.expand(kl.size(0), -1)
            kl_obj = kl_obj.sum(dim=1)

            # sum over all the dimensions, but the batch
            kl = kl.sum(dim=(1,2,3))

        h = torch.cat((z, h_det), 1)
        h = F.elu(h)

        if self.ds:
            input = F.upsample(input, scale_factor=2.)
        
        h = self.down_conv_b(h)

        return input + 0.1 * h, kl, kl_obj 

