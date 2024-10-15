# These code is adapted from https://github.com/dreamgaussian/dreamgaussian/blob/main/process.py
# The original code is licensed under the MIT License.

# python preprocess.py /home/haohe/project/ZeroShape/temp/bear_unprocessed --outdir /home/haohe/project/ZeroShape/temp/bear  --size 256 
# python preprocess.py /hpc2hdd/home/hheat/projects/gs_shape/temp/skates --outdir /hpc2hdd/home/hheat/projects/gs_shape/temp/skates
# python preprocess.py /hpc2hdd/home/hheat/projects/gs_shape/temp/yc_demo/kunkun --outdir /hpc2hdd/home/hheat/projects/gs_shape/temp/yc_demo/kunkun


import os
import glob
import sys
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import rembg
from kiui.op import recenter

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="path to image (png, jpeg, etc.)")
    parser.add_argument('--outdir', type=str, help="output path to image (png, jpeg, etc.)")
    parser.add_argument('--model', default='u2net', type=str, help="rembg model, see https://github.com/danielgatis/rembg#models")   
    opt = parser.parse_args()
    opt.recenter=False


    session = rembg.new_session(model_name=opt.model)

    if os.path.isdir(opt.path):
        print(f'[INFO] processing directory {opt.path}...')
        files = glob.glob(f'{opt.path}/*.png') + glob.glob(f'{opt.path}/*.jpeg')
    else: # isfile
        files = [opt.path]

    out_dir = opt.outdir
    out_images_dir = os.path.join(out_dir, 'images')
    out_masks_dir = os.path.join(out_dir, 'masks')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_masks_dir, exist_ok=True)
    
    for file in files:

        out_base = os.path.basename(file).split('.')[0]
        out_rgba = os.path.join(out_images_dir, out_base + '.png')
        out_mask = os.path.join(out_masks_dir, out_base + '.png')

        # load image
        print(f'[INFO] loading image {file}...')
        image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        
        # carve background
        print(f'[INFO] background removal...')
        carved_image = rembg.remove(image, session=session) # [H, W, 4]
        mask = carved_image[..., -1] > 127

        final_rgba = recenter(carved_image, mask, border_ratio=0.2)

        final_mask = (final_rgba[..., -1] > 127).astype(np.uint8) * 255
        
        # write image
        cv2.imwrite(out_rgba, final_rgba)
        cv2.imwrite(out_mask, final_mask)