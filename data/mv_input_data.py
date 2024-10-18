import torch
import torchvision.transforms.functional as torchvision_F
import os
import numpy as np
import rembg
from torchvision.transforms import Resize
from utils.util import EasyDict as edict 
from glob import glob
from PIL import Image



##### MV Input related ######
def get_1d_bounds(arr):
        nz = np.flatnonzero(arr)
        return nz[0], nz[-1]

def get_bbox_from_mask(mask, thr):
    masks_for_box = (mask > thr).astype(np.float32)
    assert masks_for_box.sum() > 0, "Empty mask!"
    x0, x1 = get_1d_bounds(masks_for_box.sum(axis=-2))
    y0, y1 = get_1d_bounds(masks_for_box.sum(axis=-1))

    return x0, y0, x1, y1

def square_crop(image, bbox, crop_ratio=1.):
    x1, y1, x2, y2 = bbox
    h, w = y2-y1, x2-x1
    yc, xc = (y1+y2)/2, (x1+x2)/2
    S = max(h, w)*1.2
    scale = S*crop_ratio
    image = torchvision_F.crop(image, top=int(yc-scale/2), left=int(xc-scale/2), height=int(scale), width=int(scale))
    return image

def preprocess_image(opt, image, bbox):
    image = square_crop(image, bbox=bbox)
    if image.size[0] != opt.input_W or image.size[1] != opt.input_H:
        image = image.resize((opt.input_W, opt.input_H))
    image = torchvision_F.to_tensor(image)
    rgb, mask = image[:3], image[3:]
    if opt.data.bgcolor is not None:
        # replace background color using mask
        rgb = rgb * mask + opt.data.bgcolor * (1 - mask)
        mask = (mask > 0.5).float()
    return rgb, mask

def get_image(opt, image_name, mask_name):
    image_fname = os.path.join(opt.data.demo_path, 'images', image_name)
    mask_fname = os.path.join(opt.data.demo_path, 'masks', mask_name)
    image = Image.open(image_fname).convert("RGB")
    mask = Image.open(mask_fname).convert("L")
    mask_np = np.array(mask)
    
    #binarize
    mask_np[mask_np <= 127] = 0
    mask_np[mask_np >= 127] = 1.0

    image = Image.merge("RGBA", (*image.split(), mask))
    bbox = get_bbox_from_mask(mask_np, 0.5)
    rgb_input_map, mask_input_map = preprocess_image(opt, image, bbox=bbox)
    return rgb_input_map, mask_input_map

def prepare_real_data(opt):
    datadir = opt.data.demo_path
    image_names = [name for name in os.listdir(os.path.join(datadir, 'images')) 
                   if name.endswith('.png') or name.endswith('.jpeg') or name.endswith('.jpg')]
    image_names.sort()
    mask_names = [name[:-4]+'.png' for name in image_names]
    len_data = len(image_names)
    folder = datadir.split('/')[-1]
    save_folder = os.path.join(opt.save_path, folder)
    os.makedirs(save_folder, exist_ok=True)
    
    data_list = []
    rgb_input_map, mask_input_map = [], []
    for i in range(len_data):
        image_name = image_names[i]
        mask_name = mask_names[i]
        _rgb_input_map, _mask_input_map = get_image(opt, image_name, mask_name)
        rgb_input_map.append(_rgb_input_map)
        mask_input_map.append(_mask_input_map)
    rgb_input_map = torch.cat([_img for _img in rgb_input_map[:opt.test_input_frames]], dim=-1)
    ref_mask_input_map = torch.stack(mask_input_map[:opt.test_input_frames], dim=0)

        
    var = edict()
    var.rgb_input_map = rgb_input_map.to(opt.device).unsqueeze(0)
    var.ref_mask_input_map = ref_mask_input_map.to(opt.device).unsqueeze(0)
    var.idx = torch.tensor([1+1]).to(opt.device).long()

    return [var], ['test'], save_folder

def prepare_gso_data(opt):
    # for demo gso
    path = opt.data.demo_path
    fnames = [f for f in glob(f'{path}/*.png')]
    _folder = path.split('/')[-1]
    ref_imgs, ref_masks = [], []
    resize_img = Resize(256, antialias=True)
    save_folder = os.path.join(opt.save_path, _folder)
    os.makedirs(save_folder, exist_ok=True)

    for fn in sorted(fnames, reverse=True):
        image = Image.open(fn)
        im_data = np.array(image.convert("RGBA"))
        norm_data = im_data / 255.0
        image = np.array(image.convert("RGB"))
        image = torchvision_F.to_tensor(image)
        mask = torchvision_F.to_tensor(norm_data[:, :, 3:4])
        bg_color = torch.ones(3, dtype=torch.float32, device=image.device)
        image = image * mask + bg_color.view(3, 1, 1) * (1 - mask)
        ref_imgs.append(image.float())
        ref_masks.append(mask)

    rgb_input_map = torch.cat([resize_img(_img) for _img in ref_imgs[:opt.test_input_frames]], dim=-1)
    ref_mask_input_map = resize_img(torch.stack(ref_masks[:opt.test_input_frames], dim=0))
    var = edict()
    var.rgb_input_map = rgb_input_map.to(opt.device).unsqueeze(0)
    var.ref_mask_input_map = ref_mask_input_map.to(opt.device).unsqueeze(0)
    var.idx = torch.tensor([1+1]).to(opt.device).long()

    return [var], ['test'], save_folder

##### MV Input related ######