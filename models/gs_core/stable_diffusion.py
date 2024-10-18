from audioop import mul
from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import StableDiffusionPipeline, DiffusionPipeline, DDPMScheduler, DDIMScheduler, EulerDiscreteScheduler, \
                      EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler, ControlNetModel, \
                      DDIMInverseScheduler, UNet2DConditionModel
from diffusers.utils.import_utils import is_xformers_available
from os.path import isfile
from pathlib import Path
import os
import random

from torchvision.utils import save_image
import torch
import torch.nn as nn


class StableDiffusion(nn.Module):
    def __init__(self, opt, get_gs_feat=False):
        super().__init__()

        self.device = opt.device
        self.precision_t = torch.float32

        if opt.model_key:
            model_key = '/hpc2hdd/home/hheat/projects/shape_ccm/pretrained/stable-diffusion-2-1-base'
        else:
            model_key = 'stabilityai/stable-diffusion-2-1'
        pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=self.precision_t, use_safetensors=True)
        self.opt = opt
        self.get_gs_feat = get_gs_feat
        pipe = pipe.to(self.device)
        pipe.enable_xformers_memory_efficient_attention()

        pipe = pipe
        self.vae = pipe.vae
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        # Freeze vae and text_encoder and set unet to trainable
        self.vae.encoder.requires_grad_(False)
        self.vae.decoder.requires_grad_(True)
        self.unet.requires_grad_(True)

        inputs = tokenizer([''], padding='max_length', max_length=tokenizer.model_max_length, truncation=True, return_tensors='pt')
        self.embeddings = text_encoder(inputs.input_ids.to(self.device))[0]
        
        self.t = torch.tensor([int(999)]).to(self.device)

        ### GS render required ######
        ### define gs hook for gs feature head ######
        self.gs_feat = None
        if self.get_gs_feat:
            # define hook
            def gs_unet_feature_hook(model, input, output):
                self.gs_feat = input[0]
            self.unet.conv_out.register_forward_hook(gs_unet_feature_hook)

        if opt.device == 0:
            print(f'[INFO] loaded stable diffusion 2.1 with xformer!')

    def forward(self,rgb_maps, register_token=None):
        rgb_maps = rgb_maps.squeeze(1)
        b,_,H,W = rgb_maps.shape # V=1 for square input, see svd for v!=1
        rgb_latent = self.encode_imgs(rgb_maps)
        if register_token != None:
            register_token = register_token.expand(b, 4, -1, -1)
            rgb_latent = torch.cat([rgb_latent, register_token], dim=-1)
        tt = self.t.expand(b).detach()
        embeddings = self.embeddings.expand(b,-1,-1).detach()
        unet_output = self.unet(rgb_latent, tt, encoder_hidden_states=embeddings).sample
        out = self.decode_ccm(unet_output)
        #stage 1 encode image[0,1]
        return out.unsqueeze(1), self.gs_feat.unsqueeze(1) if self.get_gs_feat else self.gs_feat # make it b,v,c,h,w
    
    def decode_latents(self, latents):
        target_dtype = latents.dtype
        latents = latents / self.vae.config.scaling_factor
        imgs = self.vae.decode(latents.to(self.vae.dtype)).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs.to(target_dtype)

    def decode_ccm(self, latents):
        target_dtype = latents.dtype
        latents = latents / self.vae.config.scaling_factor
        imgs = self.vae.decode(latents.to(self.vae.dtype)).sample
        imgs = imgs.clamp(-1, 1)
        return imgs.to(target_dtype)
    

    def encode_imgs(self, imgs):
        target_dtype = imgs.dtype
        # imgs: [B, 3, H, W]
        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs.to(self.vae.dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents.to(target_dtype)