import torch
import numpy as np
import os
import sys
import shutil
import utils.options as options
import utils.util_vis as util_vis
from utils.util import move_to_device
from tqdm import tqdm
from models.compute_graph.graph_ccm_stage_2_gs_square import Graph
from glob import glob

import imageio
from torchvision.utils import save_image
from einops import rearrange
 
from utils.camera import generate_circle_poses, generate_lgm_poses, get_proj_matrix, convert_to_opencv
from data.mv_input_data import prepare_real_data, prepare_gso_data
from data.diffusion_data import prepare_crm_data, prepare_imagegen_data, setup_crm_diffusion, setup_imagegen_diffusion


def gen(opt, graph, pipeline=None):
    if (opt.image_data == True) and (opt.single_input == True):
        raise # you cant do both
    if opt.image_data:
        print('[INFO] Using random image input data ...')
        data_list, name_list, load_path = prepare_real_data(opt)
    elif opt.single_input:
        if opt.crm:
            print('[INFO] Using mv diffusion from CRM...')
            data_list, name_list, load_path = prepare_crm_data(opt, pipeline)
        else:
            print('[INFO] Using MV dream...')
            data_list, name_list, load_path = prepare_imagegen_data(opt, pipeline)
    else:
        # gso demo in paper, no need to rmbg
        print('[INFO] Using demo GSO data ...')
        data_list, name_list, load_path = prepare_gso_data(opt)
    if opt.lucid_cam:
        print(f'[INFO] Using camera orbit from Lucid Dreamer.... ')
        poses = generate_circle_poses(size=150) # 150, 4, 4
    else:
        print('[INFO] Using camera orbti from LGM... ')
        poses = generate_lgm_poses() # 180, 4, 4
    proj_matrix = get_proj_matrix()
    gs_cam_view, gs_cam_view_proj, gs_cam_pos = convert_to_opencv(poses, proj_matrix)
    data_list[0].gs_cam_view = gs_cam_view.unsqueeze(0)
    data_list[0].gs_cam_view_proj = gs_cam_view_proj.unsqueeze(0)
    data_list[0].gs_cam_pos = gs_cam_pos.unsqueeze(0)
    print('==> sample data loaded from: {}'.format(load_path))

     # create the save dir
    save_folder = os.path.join(load_path, 'preds')
    if os.path.isdir(save_folder):
        shutil.rmtree(save_folder)
    os.makedirs(save_folder)
    if opt.save_frames:
        frame_folder = os.path.join(save_folder, 'frames')
        if os.path.isdir(frame_folder):
            shutil.rmtree(frame_folder)
        os.makedirs(frame_folder)
    opt.output_path = load_path

    # inference the model and save the results
    progress_bar = tqdm(data_list)
    for i, var in enumerate(progress_bar):
        # forward
        with torch.no_grad():
            var = move_to_device(var, opt.device)
            var = graph.forward(opt, var, training=False, get_loss=False, ccm_only=False)
            if 'seen_points_pred' in var:
                _filename_pred = ('ccm_input', 'seen_surface_pred')
                util_vis.warp_vis_function(opt, var.idx, _filename_pred, var.seen_points_pred, var.rgb_input_map[0], folder='preds')
            if opt.save_per_view_ply:
                _fname = 'per_view_ply'
                for i, ply in enumerate(var.seen_points_pred):
                    util_vis.vis_per_view_ply(opt, i, _fname, ply.unsqueeze(0), var.rgb_input_map[0][i:i+1], folder='preds')
            if 'pred_images' in var and opt.save_frames:
                pred_frames = var.pred_images.reshape(-1, 3, 512, 512)
                for i, img in enumerate(pred_frames):
                    # import pdb; pdb.set_trace()
                    util_vis.dump_gs_images(opt, i, 'gs_rendering', img.unsqueeze(0), None, folder='preds/frames')
            if 'opacity' in var:
                # import pdb; pdb.set_trace()
                conf_map = rearrange(var.opacity, 'b (v h w) c -> (b v) c h w', h=256, v=var.rgb_input_map.shape[1])
                fn = opt.output_path.split('/')[-1]
                _file_path = "{}/{}/{}_{}.png".format(opt.output_path, 'preds', fn, 'conf_map')
                save_image(conf_map, _file_path)
            if opt.save_video:
                fn = opt.output_path.split('/')[-1]
                frames = []
                for p in var.pred_images.reshape(-1, 3, 512, 512):
                    p = torch.clamp(p, 0.0, 1.0)
                    p = p.permute(1,2,0).detach().cpu().numpy()
                    p = (p * 255).round().astype('uint8')
                    frames.append(p)
                _file_path = "{}/{}/{}_{}.mp4".format(opt.output_path, 'preds', fn, 'video_rgb')
                imageio.mimwrite(_file_path, frames, fps=30, quality=8)
                
    print('==> results saved at folder: {}/preds'.format(opt.output_path))


def main():
    opt_cmd = options.parse_arguments(sys.argv[1:])
    opt = options.set(opt_cmd=opt_cmd, safe_check=False)
    opt.device = 0
    pipeline = None
    
    # build model
    task_ckpt = opt.yaml.split('/')[-1].split('.')[0]
    if task_ckpt != opt.task:
        raise ValueError('Detected different tasks between specified and the yaml, please double check!')

    # load checkpoint
    graph = Graph(opt).to(opt.device)
    checkpoint = torch.load(opt.ckpt, map_location=torch.device(opt.device))
    ep, it, best_val = checkpoint["epoch"], checkpoint["iter"], checkpoint["best_val"]
    print("resuming from iteration {}, best_val {:.4f}".format(it, best_val))
    graph.load_state_dict(checkpoint["graph"], strict=True)
    graph.eval()
    print('==> checkpoint loaded')
    if opt.single_input:
        if opt.crm:
            pipeline = setup_crm_diffusion()
        else:
            pipeline = setup_imagegen_diffusion()
            # mv dream should also setup here!!!!
    
    gen(opt, graph, pipeline=pipeline)

if __name__ == "__main__":
    main()