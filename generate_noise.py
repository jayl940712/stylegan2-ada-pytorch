# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os, sys
import re
import copy
from typing import List, Optional

import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy
sys.path.append('/workspace/mingjiel/pytorch-CycleGAN-and-pix2pix')
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import torch
import numpy as np

from torch.distributions.normal import Normal

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')
    
def calculate_iou(a, b):
    idx0, idx1 = a <= 127, a > 127
    a[idx0], a[idx1] = 0, 1
    idx0, idx1 = b <= 127, b > 127
    b[idx0], b[idx1] = 0, 1
    intersection_fg = (a & b).float()
    union_fg = (a | b).float()
    iou_fg = intersection_fg.sum()/union_fg.sum()
    iou_bg = (1-union_fg).sum()/(1-intersection_fg).sum()
    iou = (iou_bg + iou_fg)/2.0
    return iou_fg.item(), iou_bg.item()

def uniform_quantize(k, gradient_clip=False):
    class qfn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            if k == 32:
                out = input
            elif k == 1:
                out = torch.sign(input)
            else:
                n = float(2 ** k - 1)
                out = torch.round(input * n) / n
            return out

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            if gradient_clip:
                grad_input.clamp_(-1, 1)
            return grad_input

    return qfn().apply

def uniform_quantize(k, gradient_clip=False):
    class qfn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            if k == 32:
                out = input
            elif k == 1:
                out = torch.sign(input)
            else:
                n = float(2 ** k - 1)
                out = torch.round(input * n) / n
            return out

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            if gradient_clip:
                grad_input.clamp_(-1, 1)
            return grad_input

    return qfn().apply
    
def attack_style(z, noise_module, G, model, lr_alpha, dist_norm, loss_type, quantize_aware, device, epochs=10, gradient_clip=True):
    upsampler = torch.nn.Upsample(scale_factor=8, mode='bicubic')
    G.eval(), model.eval()
    label = torch.zeros([1, G.c_dim], device=device)
    optimizer = torch.optim.Adam([z], lr=lr_alpha)
    z.requires_grad = True
    norm_dist = Normal(torch.tensor([0.0], device=device), torch.tensor([1.0], device=device))
    quantizer = uniform_quantize(k=1, gradient_clip=gradient_clip)
    for i in range(epochs):
        optimizer.zero_grad()
        img = G(z, label, truncation_psi=1.0, noise_mode='const')
        img = upsampler(img)
        if quantize_aware:
            img = quantizer(img)
        else:
            img = model.legalize_mask(img, 0)
        img = (img + 1 ) * 0.5
        model.mask = img
        if dist_norm > 1e-5:
            loss = model.forward_uncertainty(loss_type) - dist_norm * norm_dist.log_prob(z).mean()
        else:
            loss = model.forward_uncertainty(loss_type)
        loss.backward()
        optimizer.step()
    return img

def attack_img(img, model, lr_alpha, quantize_aware, epochs, gradient_clip=True):
    optimizer = torch.optim.Adam([img], lr=lr_alpha)
    img.requires_grad = True
    original=None
    quantizer = uniform_quantize(k=1, gradient_clip=gradient_clip)
    for i in range(epochs):
        optimizer.zero_grad()
        if quantize_aware:
            x_img = quantizer(img)
        else:
            x_img = model.legalize_mask(img, 0)
        x_img = (x_img + 1) * 0.5
        model.mask = x_img
        loss = -model.forward_attack(original) 
        if original == None:
            original = model.real_resist.detach()
        loss.backward(retain_graph=True)
        optimizer.step()
    img = model.legalize_mask(img, 0)
    img = (img + 1) * 0.5
    return img
    
    
def attack_noise(z, noise_module, G, model, lr_alpha, dist_norm, quantize_aware, device, epochs=10, gradient_clip=True):
    noise, noise_block = noise_module.generate()
    upsampler = torch.nn.Upsample(scale_factor=8, mode='bicubic')
    G.eval(), model.eval()
    label = torch.zeros([1, G.c_dim], device=device)
    optimizer = torch.optim.Adam([noise], lr=lr_alpha)
    noise.requires_grad = True
    original = None
    quantizer = uniform_quantize(k=1, gradient_clip=gradient_clip)
    norm_dist = Normal(torch.tensor([0.0], device=device), torch.tensor([1.0], device=device))
    for i in range(epochs):
        optimizer.zero_grad()
        noise_block = noise_module.transform_noise(noise)
        img = G(z, label, truncation_psi=1.0, noise_mode='random', input_noise=noise_block)
        img = upsampler(img)
        if quantize_aware:
            img = quantizer(img)
        else:
            img = model.legalize_mask(img, 0)
        img = (img + 1) * 0.5
        model.mask = img
        if dist_norm > 1e-5:
            loss = -model.forward_attack(original) - dist_norm * norm_dist.log_prob(noise).mean()
        else:
            loss = -model.forward_attack(original)
        if original is None:
            original = model.real_resist.detach()
        loss.backward()
        optimizer.step()
    return img

class Noise:
    def __init__(self, block_resolutions, device='cpu'):
        self.block_resolutions = block_resolutions
        self.num = (1, )  + (2, ) * (len(self.block_resolutions) - 1)
        self.device = device
        self.size = sum([(self.block_resolutions[i] ** 2) * self.num[i] for i in range(len(self.num))])
    def generate(self, batch=1, input=None):
        if input == None:
            noise = torch.randn(batch*self.size, device=self.device)
        else:
            noise = input
            assert noise.shape == (batch*self.size, )
        block_noise = {}
        idx = 0
        for num, res in zip(self.num, self.block_resolutions):
            block_noise[f'b{res}'] = []
            for _ in range(num):
                length = batch * res * res
                cur = noise[idx: idx+length].reshape(-1, 1, res, res)
                idx += length
                block_noise[f'b{res}'].append(cur)
        return noise, block_noise
    def transform_noise(self, noise, batch=1):
        block_noise = {}
        idx = 0
        for num, res in zip(self.num, self.block_resolutions):
            block_noise[f'b{res}'] = []
            for _ in range(num):
                length = batch * res * res
                cur = noise[idx: idx+length].reshape(-1, 1, res, res)
                idx += length
                block_noise[f'b{res}'].append(cur)
        return block_noise
    
def generate_images():
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate curated MetFaces images without truncation (Fig.10 left)
    python generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate uncurated MetFaces images with truncation (Fig.12 upper left)
    python generate.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate class conditional CIFAR-10 images (Fig.17 left, Car)
    python generate.py --outdir=out --seeds=0-35 --class=1 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl

    \b
    # Render an image from projected W
    python generate.py --outdir=out --projected_w=projected_w.npz \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
    """

    truncation_psi = 1.0
    
    opt = TestOptions().parse()
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    model.eval()

    os.makedirs(opt.outdir, exist_ok=True)
    network_pkl = opt.stylegan
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project='CycleGAN-and-pix2pix', name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')
        
    noise_module = Noise(G.synthesis.block_resolutions, device)
    label = torch.zeros([1, G.c_dim], device=device)
    upsampler = torch.nn.Upsample(scale_factor=8, mode='bicubic')
    
    # Generate images.
    
    def generate_img(num, outdir):
        seeds = list(range(num)) 
        results = []
        for i, seed in enumerate(seeds):
            #print('Generating image for seed %d (%d/%d) ...' % (seed, i, len(seeds)))
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
            img = G(z, label, truncation_psi=truncation_psi, noise_mode='const')
            img = upsampler(img)
            img = (img + 1) * 0.5
            model.mask = img
            model.legalize_mask(model.mask)
            model.forward()
            _, iou_fg = model.get_F_criterion(None)
            results.append(iou_fg)
            mask_golden = (model.real_resist[0,0,:,:] * 255).to(torch.uint8)
            mask_pred = (model.real_mask)
            img_output = (img[0,0,:,:] * 255).to(torch.uint8)
            img_output = torch.cat((img_output,mask_golden), 1)
            PIL.Image.fromarray(img_output.detach().cpu().numpy(), 'L').save(f'{outdir}/seed{i:04d}.png')
        return results
            
    def attack_style_loop(num, outdir, lr_alpha, dist_norm, loss_type, quantize_aware, attack_epoch):
        seeds = list(range(num)) 
        results = []
        for i, seed in enumerate(seeds):
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
            img = attack_style(z, noise_module, G, model, lr_alpha, dist_norm, loss_type, quantize_aware, device=device, epochs=attack_epoch)
            model.mask = img
            model.forward()
            _, iou_fg = model.get_F_criterion(None)
            results.append(iou_fg)
            mask_output = (model.real_mask[0,0,:,:] * 255).to(torch.uint8)
            mask_golden = (model.real_resist[0,0,:,:] * 255).to(torch.uint8)
            img_output = (img[0,0,:,:] * 255).to(torch.uint8)
            img_output = torch.cat((img_output,mask_golden, mask_output), 1)
            PIL.Image.fromarray(img_output.detach().cpu().numpy(), 'L').save(f'{outdir}/seed{i:04d}.png')
        return results

    def attack_img_loop(num, outdir, lr_alpha, quantize_aware, attack_epoch):
        seeds = list(range(num)) 
        results = []
        for i, seed in enumerate(seeds):
            print('Generating image for seed %d (%d/%d) ...' % (seed, i, len(seeds)))
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
            #noise, noise_block = noise_module.generate()
            img = G(z, label, truncation_psi=truncation_psi, noise_mode='const')
            img = upsampler(img).detach()
            print(img)
            img = attack_img(img, model, lr_alpha, quantize_aware, epochs=attack_epoch)
            model.mask = img
            model.forward()
            _, iou_fg = model.get_F_criterion(None)
            results.append(iou_fg)
            mask_output = (model.real_mask[0,0,:,:] * 255).to(torch.uint8)
            mask_golden = (model.real_resist[0,0,:,:] * 255).to(torch.uint8)
            img_output = (img[0,0,:,:] * 255).to(torch.uint8)
            img_output = torch.cat((img_output,mask_golden, mask_output), 1) 
            PIL.Image.fromarray(img_output.detach().cpu().numpy(), 'L').save(f'{outdir}/seed{i:04d}.png')
        return results
            
    def attack_noise_loop(num, outdir, lr_alpha, dist_norm, quantize_aware, attack_epoch):
        seeds = list(range(num)) 
        results = []
        for i, seed in enumerate(seeds):
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
            img = attack_noise(z, noise_module, G, model, lr_alpha, dist_norm, quantize_aware, device=device, epochs=attack_epoch)
            model.mask = img
            model.forward()
            _, iou_fg = model.get_F_criterion(None)
            results.append(iou_fg)
            results.append(b)
            img_output = (img[0,0,:,:] * 255).to(torch.uint8)
            mask_golden = (model.real_resist[0,0,:,:] * 255).to(torch.uint8)
            mask_output = (model.real_mask[0,0,:,:] * 255).to(torch.uint8)
            img_output = torch.cat((img_output,mask_golden, mask_output), 1)
            PIL.Image.fromarray(img_output.detach().cpu().numpy(), 'L').save(f'{outdir}/seed{i:04d}.png')
        return results
      
    if opt.aug_type == 'random':
        results = generate_img(opt.num_gen, opt.outdir)
    elif opt.aug_type == 'style':
        results = attack_style_loop(opt.num_gen, opt.outdir, opt.lr_alpha, opt.dist_norm, opt.loss_type, opt.quantize_aware, opt.attack_epoch)
    elif opt.aug_type == 'noise':
        results = attack_noise_loop(opt.num_gen, opt.outdir, opt.lr_alpha, opt.dist_norm, opt.quantize_aware, opt.attack_epoch)
    else:
        results = attack_img(opt.num_gen, opt.outdir, opt.lr_alpha, opt.quantize_aware, opt.attack_epoch)
    
    print(sum(results) / len(results))



#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
