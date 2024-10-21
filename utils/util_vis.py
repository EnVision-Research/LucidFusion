import numpy as np
import os
import torch
import torchvision
import torchvision.transforms.functional as torchvision_F
import matplotlib.pyplot as plt
import PIL
import PIL.ImageDraw
from PIL import Image, ImageFont
import trimesh
import cv2
import copy
import base64
import io
import imageio
from utils.camera import vis_ccm_map
from torchvision.utils import save_image

def dump_ccm(opt, idx, name, ccm, rgb, folder='dump'):
    file_path = "{}/{}/{}_{}.png".format(opt.output_path, folder, idx.item(), name)
    vis_ccm_map(ccm, opt.H, opt.W, rgb, file_path)

def warp_vis_function(opt, idx, filename, vis_ccm, vis_rgb, folder):
    '''
    opt: opt file
    idx: int
    filename: tuple, file name to be saved
    vis_ccm: tensor, [V, N, 3], V is number of views
    vis_rgb: tensor, [V, 3, H, W]
    folder: str, folder to be saved

    Save two files, named {filename.png} and {filename.ply}.
    '''
    assert len(vis_rgb.shape) == 4
    assert len(vis_ccm.shape) == 3
    V, _, H, W = vis_rgb.shape
    if not torch.is_tensor(idx):
        idx = torch.tensor(idx).unsqueeze(0)
    dump_ccm(opt, idx, filename[0], vis_ccm.permute(0,2,1).reshape(V, 3, H, W), vis_rgb, folder=folder) #non square
    dump_pointclouds(opt, idx, filename[1], vis_ccm.reshape(1,-1,3), vis_rgb.permute(0, 2, 3, 1).reshape(1, -1, 3), folder=folder)

def vis_per_view_ply(opt, idx, filename, vis_ccm, vis_rgb, folder):
    '''
    opt: opt file
    idx: int
    filename: tuple, file name to be saved
    vis_ccm: tensor, [V, N, 3], V is number of views
    vis_rgb: tensor, [V, 3, H, W]
    folder: str, folder to be saved

    Save two files, named {filename.png} and {filename.ply}.
    '''
    assert len(vis_rgb.shape) == 4
    assert len(vis_ccm.shape) == 3
    V, _, H, W = vis_rgb.shape
    if not torch.is_tensor(idx):
        idx = torch.tensor(idx).unsqueeze(0)
    dump_pointclouds(opt, idx, filename, vis_ccm, vis_rgb.permute(0, 2, 3, 1).reshape(1, -1, 3), folder=folder)

def dump_gs_images(opt, idx, name, gs_pred, gs_gt, folder='dump'):
    '''
    idx: int
    name: str, filename to be saved
    gs_pred: V, 3, H, W
    gs_gt: V, 3, H, W
    folder: str, folder to be saved

    Save cat (at -2 dim) GT and pred image.
    '''
    if not torch.is_tensor(idx):
        idx = torch.tensor(idx)
    file_path = "{}/{}/{}_{}.png".format(opt.output_path, folder, idx.item(), name)
    if gs_gt == None:
        vis = gs_pred
    else:
        assert gs_gt.shape[-1] == gs_pred.shape[-1]
        vis = torch.cat([gs_gt, gs_pred], dim=-2)
    save_image(vis, file_path)

def dump_seen_surface(opt, idx, obj_name, img_name, seen_projs, folder='dump'):
    # seen_proj: [B, H, W, 3]
    for i, seen_proj in zip(idx, seen_projs):
        out_folder = "{}/{}".format(opt.output_path, folder)
        img_fname = "{}_{}.png".format(i, img_name)
        create_seen_surface(i, img_fname, seen_proj, out_folder, obj_name)

# https://github.com/princeton-vl/oasis/blob/master/utils/vis_mesh.py
def create_seen_surface(sample_ID, img_path, XYZ, output_folder, obj_name, connect_thres=0.005):
    height, width = XYZ.shape[:2]
    XYZ_to_idx = {}
    idx = 1
    with open("{}/{}_{}.mtl".format(output_folder, sample_ID, obj_name), "w") as f:
        f.write("newmtl material_0\n")
        f.write("Ka 0.200000 0.200000 0.200000\n")
        f.write("Kd 0.752941 0.752941 0.752941\n")
        f.write("Ks 1.000000 1.000000 1.000000\n")
        f.write("Tr 1.000000\n")
        f.write("illum 2\n")
        f.write("Ns 0.000000\n")
        f.write("map_Ka %s\n" % img_path)
        f.write("map_Kd %s\n" % img_path)

    with open("{}/{}_{}.obj".format(output_folder, sample_ID, obj_name), "w") as f:
        f.write("mtllib {}_{}.mtl\n".format(sample_ID, obj_name))
        for y in range(height):
            for x in range(width):
                if XYZ[y][x][2] > 0:
                    XYZ_to_idx[(y, x)] = idx
                    idx += 1
                    f.write("v %.4f %.4f %.4f\n" % (XYZ[y][x][0], XYZ[y][x][1], XYZ[y][x][2]))
                    f.write("vt %.8f %.8f\n" % ( float(x) / float(width), 1.0 - float(y) / float(height)))
        f.write("usemtl material_0\n")
        for y in range(height-1):
            for x in range(width-1):
                if XYZ[y][x][2] > 0 and XYZ[y][x+1][2] > 0 and XYZ[y+1][x][2] > 0:
                    # if close enough, connect vertices to form a face
                    if torch.norm(XYZ[y][x] - XYZ[y][x+1]).item() < connect_thres and torch.norm(XYZ[y][x] - XYZ[y+1][x]).item() < connect_thres:
                        f.write("f %d/%d %d/%d %d/%d\n" % (XYZ_to_idx[(y, x)], XYZ_to_idx[(y, x)], XYZ_to_idx[(y, x+1)], XYZ_to_idx[(y, x+1)], XYZ_to_idx[(y+1, x)], XYZ_to_idx[(y+1, x)]))
                if XYZ[y][x+1][2] > 0 and XYZ[y+1][x+1][2] > 0 and XYZ[y+1][x][2] > 0:
                    if torch.norm(XYZ[y][x+1] - XYZ[y+1][x+1]).item() < connect_thres and torch.norm(XYZ[y][x+1] - XYZ[y+1][x]).item() < connect_thres:
                        f.write("f %d/%d %d/%d %d/%d\n" % (XYZ_to_idx[(y, x+1)], XYZ_to_idx[(y, x+1)], XYZ_to_idx[(y+1, x+1)], XYZ_to_idx[(y+1, x+1)], XYZ_to_idx[(y+1, x)], XYZ_to_idx[(y+1, x)]))

def dump_pointclouds_compare(opt, idx, name, preds, gts, folder='dump'):
    for i in range(len(idx)):
        pred = preds[i].cpu().numpy()   # [N1, 3]
        gt = gts[i].cpu().numpy()   # [N2, 3]
        color_pred = np.zeros(pred.shape).astype(np.uint8)
        color_pred[:, 0] = 255
        color_gt = np.zeros(gt.shape).astype(np.uint8)
        color_gt[:, 1] = 255
        pc_vertices = np.vstack([pred, gt])
        colors = np.vstack([color_pred, color_gt])
        pc_color = trimesh.points.PointCloud(vertices=pc_vertices, colors=colors)
        fname = "{}/{}/{}_{}.ply".format(opt.output_path, folder, idx[i], name)
        pc_color.export(fname)

def dump_pointclouds(opt, idx, name, pcs, colors, folder='dump', colormap='jet'):
    for i, pc, color in zip(idx, pcs, colors):
        pc = pc.cpu().numpy()   # [N, 3]
        color = color.cpu().numpy()   # [N, 3] or [N, 1]
        # convert scalar color to rgb with colormap
        if color.shape[1] == 1:
            # single channel color in numpy between [0, 1] to rgb
            color = plt.get_cmap(colormap)(color[:, 0])
            color = (color * 255).astype(np.uint8)
        pc_color = trimesh.points.PointCloud(vertices=pc, colors=color)
        fname = "{}/{}/{}_{}.ply".format(opt.output_path, folder, i, name)
        pc_color.export(fname)

@torch.no_grad()
def vis_pointcloud(opt, vis, step, split, pred, GT=None):
    win_name = "{0}/{1}".format(opt.group, opt.name)
    pred, GT = pred.cpu().numpy(), GT.cpu().numpy()
    for i in range(opt.visdom.num_samples):
        # prediction
        data = [dict(
            type="scatter3d",
            x=[float(n) for n in points[i, :opt.visdom.num_points, 0]],
            y=[float(n) for n in points[i, :opt.visdom.num_points, 1]],
            z=[float(n) for n in points[i, :opt.visdom.num_points, 2]],
            mode="markers",
            marker=dict(
                color=color,
                size=1,
            ),
        ) for points, color in zip([pred, GT], ["blue", "magenta"])]
        vis._send(dict(
            data=data,
            win="{0} #{1}".format(split, i),
            eid="{0}/{1}".format(opt.group, opt.name),
            layout=dict(
                title="{0} #{1} ({2})".format(split, i, step),
                autosize=True,
                margin=dict(l=30, r=30, b=30, t=30, ),
                showlegend=False,
                yaxis=dict(
                    scaleanchor="x",
                    scaleratio=1,
                )
            ),
            opts=dict(title="{0} #{1} ({2})".format(win_name, i, step), ),
        ))
