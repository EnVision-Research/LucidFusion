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

# from torchvision.utils import save_image
import torch
import torch.nn as nn
from einops import rearrange


class StableDiffusion(nn.Module):
    def __init__(self, opt, get_gs_feat=False): # Testing, merge feat to opt later
        super().__init__()

        self.device = opt.device
        self.precision_t = torch.float32 #torch.float16 if opt.optim.amp else torch.float32
        model_key = f'{opt.model_key}'
        pipe = DiffusionPipeline.from_pretrained(model_key, torch_dtype=self.precision_t)
        self.opt = opt
        self.get_gs_feat = get_gs_feat
        
        pipe = pipe.to(self.device)
        pipe.enable_xformers_memory_efficient_attention()


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
        # self.embeddings = text_encoder(inputs.input_ids)[0]
        
        self.t = torch.tensor([int(999)]).to(self.device)
        # self.t = torch.tensor([int(999)])

        ### GS render required ######
        ### define gs hook for gs feature head ######
        self.gs_feat = None
        if self.get_gs_feat:
            # define hook
            def gs_unet_feature_hook(model, input, output):
                self.gs_feat = input[0]
            self.unet.conv_out.register_forward_hook(gs_unet_feature_hook)


        print(f'[INFO] loaded stable video diffusion!')

    def forward(self, rgb_maps, register_token=None):
        b,V,_,H,W = rgb_maps.shape
        rgb_maps = rearrange(rgb_maps, 'b v c h w -> (b v) c h w')
        rgb_latent = self.encode_imgs(rgb_maps)
        if register_token != None:
            register_token = register_token.expand(b*V, 4, -1, -1).contiguous()
            rgb_latent = torch.cat([rgb_latent, register_token], dim=-2)
        rgb_latent = rearrange(rgb_latent, "(b v) c h w -> b c v h w", v=V)
        tt = self.t.expand(b).contiguous().detach()
        embeddings = self.embeddings.expand(b,-1,-1).detach()
        unet_output = self.unet(rgb_latent, tt, encoder_hidden_states=embeddings).sample
        unet_output = rearrange(unet_output, "b c f h w -> (b f) c h w")
        out = self.decode_ccm(unet_output)
        out = rearrange(out, "(b v) c h w -> b v c h w", v=V)
        if self.gs_feat != None:
            self.gs_feat = rearrange(self.gs_feat, "(b v) c h w -> b v c h w", v=V)
        return out, self.gs_feat
    
    def decode_latents(self, latents):
        target_dtype = latents.dtype
        latents = latents / self.vae.config.scaling_factor
        imgs = self.vae.decode(latents.to(self.vae.dtype)).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs.to(target_dtype)


    def decode_ccm(self, latents):
        # target_dtype = latents.dtype
        latents = latents / self.vae.config.scaling_factor
        imgs = self.vae.decode(latents).sample
        imgs = imgs.clamp(-1, 1)
        return imgs
    

    def encode_imgs(self, imgs):
        # target_dtype = imgs.dtype
        # imgs: [B, V, 3, H, W]
        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents