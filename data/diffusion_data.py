import torch
import torchvision.transforms.functional as torchvision_F

import os
import numpy as np

from omegaconf import OmegaConf
from CRM.pipelines import TwoStagePipeline
from huggingface_hub import hf_hub_download

from mvdream.pipeline_mvdream import MVDreamPipeline

from torchvision.transforms import Resize
from utils.util import EasyDict as edict 
from glob import glob
import rembg
import kiui
from kiui.op import recenter


from PIL import Image


##### CRM related ######

def setup_crm_diffusion():
    stage1_config = OmegaConf.load("/hpc2hdd/home/hheat/projects/Lucidfusion/CRM/configs/nf7_v3_SNR_rd_size_stroke.yaml").config
    stage2_config = OmegaConf.load("/hpc2hdd/home/hheat/projects/Lucidfusion/CRM/configs/stage2-v2-snr.yaml").config
    stage2_sampler_config = stage2_config.sampler
    stage1_sampler_config = stage1_config.sampler

    stage1_model_config = stage1_config.models
    stage2_model_config = stage2_config.models

    xyz_path = hf_hub_download(repo_id="Zhengyi/CRM", filename="ccm-diffusion.pth")
    pixel_path = hf_hub_download(repo_id="Zhengyi/CRM", filename="pixel-diffusion.pth")
    stage1_model_config.resume = pixel_path
    stage2_model_config.resume = xyz_path

    pipeline = TwoStagePipeline(
        stage1_model_config,
        stage2_model_config,
        stage1_sampler_config,
        stage2_sampler_config,
    )
    return pipeline

def expand_to_square(image, bg_color=(0, 0, 0, 0)):
    # expand image to 1:1
    width, height = image.size
    if width == height:
        return image
    new_size = (max(width, height), max(width, height))
    new_image = Image.new("RGBA", new_size, bg_color)
    paste_position = ((new_size[0] - width) // 2, (new_size[1] - height) // 2)
    new_image.paste(image, paste_position)
    return new_image

def remove_background(
    image: Image.Image,
    rembg_session = None,
    force: bool = False,
    **rembg_kwargs,
) -> Image.Image:
    do_remove = True
    if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
        # explain why current do not rm bg
        print("alhpa channl not enpty, skip remove background, using alpha channel as mask")
        background = Image.new("RGBA", image.size, (0, 0, 0, 0))
        image = Image.alpha_composite(background, image)
        do_remove = False
    do_remove = do_remove or force
    if do_remove:
        image = rembg.remove(image, session=rembg_session, **rembg_kwargs)
    return image

def do_resize_content(original_image: Image, scale_rate):
    # resize image content wile retain the original image size
    if scale_rate != 1:
        # Calculate the new size after rescaling
        new_size = tuple(int(dim * scale_rate) for dim in original_image.size)
        # Resize the image while maintaining the aspect ratio
        resized_image = original_image.resize(new_size)
        # Create a new image with the original size and black background
        padded_image = Image.new("RGBA", original_image.size, (0, 0, 0, 0))
        paste_position = ((original_image.width - resized_image.width) // 2, (original_image.height - resized_image.height) // 2)
        padded_image.paste(resized_image, paste_position)
        return padded_image
    else:
        return original_image

def add_background(image, bg_color=(255, 255, 255)):
    # given an RGBA image, alpha channel is used as mask to add background color
    background = Image.new("RGBA", image.size, bg_color)
    return Image.alpha_composite(background, image)


def crm_preprocess_image(image, background_choice, foreground_ratio, backgroud_color, rembg_session):
    """
    input image is a pil image in RGBA, return RGB image
    """
    print(background_choice)
    if background_choice == "Alpha as mask":
        background = Image.new("RGBA", image.size, (0, 0, 0, 0))
        image = Image.alpha_composite(background, image)
    else:
        image = remove_background(image, rembg_session, force_remove=True)
    image = do_resize_content(image, foreground_ratio)
    image = expand_to_square(image)
    image = add_background(image, backgroud_color)
    return image.convert("RGB")

def prepare_crm_data(opt, pipeline=None):
    if pipeline == None:
        raise
    path = opt.data.demo_path
    # save_path = '/hpc2hdd/home/hheat/projects/gs_shape/temp/mv_gen'
    folder = path.split('/')[-1].split('.')[0]
    save_folder = os.path.join(opt.save_path, folder)
    os.makedirs(save_folder, exist_ok=True)
    rembg_session = rembg.new_session()
    img = Image.open(path)

    img = crm_preprocess_image(img, 'remove', 1.0, (127, 127, 127),rembg_session)
    rt_dict = pipeline(img, scale=5.0, step=50)
    mv_image = rt_dict["stage1_images"]
    mv_image = np.stack(mv_image, 0)
    mv_image = mv_image[(5,4,3,2,1,0),...] # last pic to first, crm diffusion!
    imgs = []
    masks = []
    for img in mv_image:
        _carved_image = rembg.remove(img, session=rembg_session) # [H, W, 4]
        _mask = _carved_image[..., -1] > 127
        _carved_image = recenter(_carved_image, _mask, border_ratio=0.2)
        _carved_image = _carved_image.astype(np.float32) / 255.0
        _mask = _carved_image[..., 3:4] # h w 1
        _carved_image = _carved_image[..., :3] * _carved_image[..., 3:4] + (1 - _carved_image[..., 3:4])
        imgs.append(_carved_image)
        masks.append(_mask)
    gen_image = np.stack(imgs, axis=0)
    masks = np.stack(masks, axis=0)

    # We dont use the top and bottom for fair comparison with LGM
    gen_image = gen_image[(0,2,3,5), ...]
    masks = masks[(0,2,3,5), ...]

    input_image = torch.from_numpy(gen_image).permute(0, 3, 1, 2).float().to(opt.device) # [4, 3, 256, 256]
    mask_input_map = torch.from_numpy(masks).permute(0, 3, 1, 2).float().to(opt.device) # [4, 1, 256, 256]
    var = edict()
    rgb_input_map = torch.cat(torch.chunk(input_image, 4, 0), dim=-1) #mv dream fix 4 views from LGM
    var.rgb_input_map = rgb_input_map
    var.ref_mask_input_map = mask_input_map.unsqueeze(0)
    var.idx = torch.tensor([1+1]).to(opt.device).long()

    return [var], ['test'], save_folder

##### CRM related ######

##### Image Gen related ######

def setup_imagegen_diffusion():
    # load image dream
    pipeline = MVDreamPipeline.from_pretrained(
        "ashawkey/imagedream-ipmv-diffusers", # remote weights
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    return pipeline

##### Image Gen related ######


def prepare_imagegen_data(opt, pipe=None):
    if pipe == None:
        raise
    path = opt.data.demo_path
    folder = path.split('/')[-1].split('.')[0]
    save_folder = os.path.join(opt.save_path, folder)
    os.makedirs(save_folder, exist_ok=True)
    
    input_image = kiui.read_image(path, mode='uint8')
    resize_img = Resize(256, antialias=True)
    input_image = torch.from_numpy(input_image).permute(2,0,1)
    input_image = resize_img(input_image)
    input_image = input_image.permute(1,2,0).numpy()

    bg_remover = rembg.new_session()
    carved_image = rembg.remove(input_image, session=bg_remover) # [H, W, 4]
    mask = carved_image[..., -1] > 127

    # recenter
    image = recenter(carved_image, mask, border_ratio=0.2)
    
    # generate mv
    image = image.astype(np.float32) / 255.0
    

    # rgba to rgb white bg
    assert image.shape[-1] == 4
    mask = (image[..., 3:4] > 127).astype(np.uint8) * 255
    image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])

    mv_image = pipe('', image, guidance_scale=5.0, num_inference_steps=30, elevation=0)
    imgs = []
    masks = []
    for img in mv_image:
        _carved_image = rembg.remove((img * 255).astype('uint8'), session=bg_remover) # [H, W, 4]
        _mask = _carved_image[..., -1] > 127
        _carved_image = recenter(_carved_image, _mask, border_ratio=0.2)
        _carved_image = _carved_image.astype(np.float32) / 255.0
        _mask = _carved_image[..., 3:4] # h w 1
        _carved_image = _carved_image[..., :3] * _carved_image[..., 3:4] + (1 - _carved_image[..., 3:4])
        imgs.append(_carved_image)
        masks.append(_mask)
    # mv_image = np.stack([mv_image[1], mv_image[2], mv_image[3], mv_image[0]], axis=0) # [4, 256, 256, 3], float32
    # input_image = torch.from_numpy(mv_image).permute(0, 3, 1, 2).float().to(opt.device) # [4, 3, 256, 256]
    gen_image = np.stack([imgs[1], imgs[2], imgs[3], imgs[0]], axis=0)
    masks = np.stack([masks[1], masks[2], masks[3], masks[0]], axis=0)

    input_image = torch.from_numpy(gen_image).permute(0, 3, 1, 2).float().to(opt.device) # [4, 3, 256, 256]
    mask_input_map = torch.from_numpy(masks).permute(0, 3, 1, 2).float().to(opt.device) # [4, 1, 256, 256]
    var = edict()
    rgb_input_map = torch.cat(torch.chunk(input_image, 4, 0), dim=-1) #mv dream fix 4 views from LGM
    var.rgb_input_map = rgb_input_map
    var.ref_mask_input_map = mask_input_map.unsqueeze(0)
    var.idx = torch.tensor([1+1]).to(opt.device).long()

    return [var], ['test'], save_folder


