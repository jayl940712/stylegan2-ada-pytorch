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
sys.path.append('/workspace/pytorch-CycleGAN-and-pix2pix')
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
    
def attack_style(z, noise_module, G, model, device, epochs=100):
    upsampler = torch.nn.Upsample(scale_factor=8, mode='bicubic')
    G.eval(), model.eval()
    label = torch.zeros([1, G.c_dim], device=device)
    optimizer = torch.optim.Adam([z], lr=0.1)
    z.requires_grad = True
    norm_dist = Normal(torch.tensor([0.0], device=device), torch.tensor([1.0], device=device))
    for i in range(epochs):
        optimizer.zero_grad()
        img = G(z, label, truncation_psi=1.0, noise_mode='const')
        img = upsampler(img)
        img = (img.clamp(-1, 1) + 1) * 0.5
        model.real_high_res = img
        loss = model.forward_uncertainty() - 1.0 * norm_dist.log_prob(z).mean()
        #loss = -model.forward_attack(None)
        tot_loss = loss
        tot_loss.backward()
        optimizer.step()
        #print(i, loss.item())#, noise.grad)
    return img

def attack_img(img, model, epochs=100):
    optimizer = torch.optim.Adam([img], lr=0.01)
    img.requires_grad = True
    original=None
    #norm_dist = Normal(torch.tensor([0.0], device=device), torch.tensor([1.0], device=device))
    for i in range(epochs):
        optimizer.zero_grad()
        x_img = (img.clamp(-1, 1) + 1) * 0.5
        model.real_high_res = x_img
        loss = -model.forward_attack(original) #- 100 * norm_dist.log_prob(z).mean()
        #loss = model.forward_uncertainty()
        if original == None:
            original = model.real_mask_img.detach()
        tot_loss = loss
        tot_loss.backward(retain_graph=True)
        optimizer.step()
        #print(i, loss.item())#, noise.grad)
    return img
    
    
def attack_noise(z, noise_module, G, model, device, epochs=100):
    noise, noise_block = noise_module.generate()
    initial_noise_block = copy.deepcopy(noise_block)
    upsampler = torch.nn.Upsample(scale_factor=8, mode='bicubic')
    G.eval(), model.eval()
    label = torch.zeros([1, G.c_dim], device=device)
    optimizer = torch.optim.Adam([noise], lr=0.1)
    noise.requires_grad = True
    original = None
    norm_dist = Normal(torch.tensor([0.0], device=device), torch.tensor([1.0], device=device))
    for i in range(epochs):
        optimizer.zero_grad()
        noise_block = noise_module.transform_noise(noise)
        img = G(z, label, truncation_psi=1.0, noise_mode='random', input_noise=noise_block)
        img = upsampler(img)
        img = (img.clamp(-1, 1) + 1) * 0.5
        model.real_high_res = img
        loss = -model.forward_attack(original) - 1.0 * norm_dist.log_prob(noise).mean()
        if original is None:
            original = model.real_mask_img.detach()
        #loss = model.forward_uncertainty() - 100 * norm_dist.log_prob(z).mean()
        tot_loss = loss 
        tot_loss.backward()
        optimizer.step()
        #print(i, loss.item())#, noise.grad)
    #print(noise.max())
    return initial_noise_block, noise_block

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

    outdir = './out'
    #seeds = [0,1,2,3,4,5,6,7,8,9]
    seeds = [0,1,2]
    truncation_psi = 1.0
    noise_mode = 'random'
    network_pkl = './stylegan_model/network-snapshot-025000.pkl'
    
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
        
    #label = torch.zeros([1, G.c_dim], device=device)

    os.makedirs(outdir, exist_ok=True)
    
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project='CycleGAN-and-pix2pix', name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')
        
    noise_module = Noise(G.synthesis.block_resolutions, device)
    label = torch.zeros([1, G.c_dim], device=device)
    upsampler = torch.nn.Upsample(scale_factor=8, mode='bicubic')
    
    # Generate images.
    
    def generate_img(num):
        seeds = list(range(num)) 
        for i, seed in enumerate(seeds):
            print('Generating image for seed %d (%d/%d) ...' % (seed, i, len(seeds)))
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
            img = G(z, label, truncation_psi=truncation_psi, noise_mode='const')
            img = upsampler(img)
            img_ori = (img.clamp(-1, 1) + 1) * 0.5
            if img_ori.sum() < 256 * 256 // 100:
                print("Image too small skipped.")
                continue
            img_output_ori = (img_ori[0,0,:,:] * 255).to(torch.uint8)
            PIL.Image.fromarray(img_output_ori.detach().cpu().numpy(), 'L').save(f'{outdir}/seed{i:04d}_img_ori.png')
            
    def attack_style_loop(num):
        seeds = list(range(num)) 
        for i, seed in enumerate(seeds):
            print('Generating image for seed %d (%d/%d) ...' % (seed, i, len(seeds)))
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
            img_ori = attack_style(z, noise_module, G, model, device=device, epochs=100)

            #noise, noise_block = noise_module.generate()
            #img = G(z, label, truncation_psi=truncation_psi, noise_mode='const')
            #img = upsampler(img)
            #img_ori = (img.clamp(-1, 1) + 1) * 0.5
            model.real_high_res = img_ori
            model.forward_F()
            a, b = model.get_F_criterion(None)
            mask_output_ori = (model.real_mask[0,0,:,:] * 255).to(torch.uint8)
            mask_golden_ori = (model.real_resist[0,0,:,:] * 255).to(torch.uint8)
            print(a, b, "attac")
            img_output_ori = (img_ori[0,0,:,:] * 255).to(torch.uint8)
            PIL.Image.fromarray(img_output_ori.detach().cpu().numpy(), 'L').save(f'{outdir}/seed{i:04d}_img_ori.png')
            #PIL.Image.fromarray(mask_output_ori.detach().cpu().numpy(), 'L').save(f'{outdir}/seed{i:04d}_ori.png')
            #PIL.Image.fromarray(mask_golden_ori.detach().cpu().numpy(), 'L').save(f'{outdir}/seed{i:04d}_golden_ori.png')
        
    def attack_img_loop(num):
        seeds = list(range(num)) 
        for i, seed in enumerate(seeds):
            print('Generating image for seed %d (%d/%d) ...' % (seed, i, len(seeds)))
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
            #noise, noise_block = noise_module.generate()
            img = G(z, label, truncation_psi=truncation_psi, noise_mode='const')
            img = upsampler(img).detach()
            img = attack_img(img, model, epochs=20)
            img_ori = (img.clamp(-1, 1) + 1) * 0.5
            model.real_high_res = img_ori
            model.forward_F()
            a, b = model.get_F_criterion(None)
            mask_output_ori = (model.real_mask[0,0,:,:] * 255).to(torch.uint8)
            mask_golden_ori = (model.real_resist[0,0,:,:] * 255).to(torch.uint8)
            print(a, b, "attac")
            img_output_ori = (img_ori[0,0,:,:] * 255).to(torch.uint8)
            PIL.Image.fromarray(img_output_ori.detach().cpu().numpy(), 'L').save(f'{outdir}/seed{i:04d}_img_ori.png')
            PIL.Image.fromarray(mask_output_ori.detach().cpu().numpy(), 'L').save(f'{outdir}/seed{i:04d}_ori.png')
            PIL.Image.fromarray(mask_golden_ori.detach().cpu().numpy(), 'L').save(f'{outdir}/seed{i:04d}_golden_ori.png')
            
    # Generate images.
    def attack_noise_loop(num):
        seeds = list(range(num)) 
        for i, seed in enumerate(seeds):
            print('Generating image for seed %d (%d/%d) ...' % (seed, i, len(seeds)))
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
            initial_noise_block, noise_block = attack_noise(z, noise_module, G, model, device=device, epochs=100)

            #noise, noise_block = noise_module.generate()
            img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode, input_noise=initial_noise_block)
            img = upsampler(img)
            img_ori = (img.clamp(-1, 1) + 1) * 0.5
            model.real_high_res = img_ori
            model.forward_F()
            a, b = model.get_F_criterion(None)
            mask_output_ori = (model.real_mask[0,0,:,:] * 255).to(torch.uint8)
            mask_golden_ori = (model.real_resist[0,0,:,:] * 255).to(torch.uint8)
            print(a, b, "init")
            img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode, input_noise=noise_block)
            img = upsampler(img)
            img = (img.clamp(-1, 1) + 1) * 0.5
            model.real_high_res = img
            model.forward_F()
            a, b = model.get_F_criterion(None)
            print(a, b, "attack")
            img_output = (img[0,0,:,:] * 255).to(torch.uint8)
            img_output_ori = (img_ori[0,0,:,:] * 255).to(torch.uint8)
            mask_output = (model.real_mask[0,0,:,:] * 255).to(torch.uint8)
            mask_golden = (model.real_resist[0,0,:,:] * 255).to(torch.uint8)
            
            #save img
            PIL.Image.fromarray(img_output.detach().cpu().numpy(), 'L').save(f'{outdir}/seed{i:04d}_img.png')
            PIL.Image.fromarray(img_output_ori.detach().cpu().numpy(), 'L').save(f'{outdir}/seed{i:04d}_img_ori.png')
            PIL.Image.fromarray(mask_output.detach().cpu().numpy(), 'L').save(f'{outdir}/seed{i:04d}.png')
            PIL.Image.fromarray(mask_output_ori.detach().cpu().numpy(), 'L').save(f'{outdir}/seed{i:04d}_ori.png')
            PIL.Image.fromarray(mask_golden.detach().cpu().numpy(), 'L').save(f'{outdir}/seed{i:04d}_golden.png')
            PIL.Image.fromarray(mask_golden_ori.detach().cpu().numpy(), 'L').save(f'{outdir}/seed{i:04d}_golden_ori.png')
                    
            # get scores
            fg, bg = calculate_iou(img_output, img_output_ori)
            print(fg, bg, "img diff")
            fg, bg = calculate_iou(mask_golden, mask_golden_ori)
            print(fg, bg, "golden diff")
            fg, bg = calculate_iou(mask_output, mask_output_ori)
            print(fg, bg, "predict diff")
      
    generate_img(2000)
    #attack_img_loop(50)
    #attack_style_loop(2000)
    #attack_noise_loop(10)



#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
