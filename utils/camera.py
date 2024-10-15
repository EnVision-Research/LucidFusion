# partially from https://github.com/chenhsuanlin/signed-distance-SRN

import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from PIL import Image

import kiui
from kiui.cam import orbit_camera
from tqdm import tqdm


class Pose():
    # a pose class with util methods
    def __call__(self, R=None, t=None):
        assert(R is not None or t is not None)
        if R is None:
            if not isinstance(t, torch.Tensor): t = torch.tensor(t)
            R = torch.eye(3, device=t.device).repeat(*t.shape[:-1], 1, 1)
        elif t is None:
            if not isinstance(R, torch.Tensor): R = torch.tensor(R)
            t = torch.zeros(R.shape[:-1], device=R.device)
        else:
            if not isinstance(R, torch.Tensor): R = torch.tensor(R)
            if not isinstance(t, torch.Tensor): t = torch.tensor(t)
        assert(R.shape[:-1]==t.shape and R.shape[-2:]==(3, 3))
        R = R.float()
        t = t.float()
        pose = torch.cat([R, t[..., None]], dim=-1) # [..., 3, 4]
        assert(pose.shape[-2:]==(3, 4))
        return pose

    def invert(self, pose, use_inverse=False):
        R, t = pose[..., :3], pose[..., 3:]
        R_inv = R.inverse() if use_inverse else R.transpose(-1, -2)
        t_inv = (-R_inv@t)[..., 0]
        pose_inv = self(R=R_inv, t=t_inv)
        return pose_inv

    def compose(self, pose_list):
        # pose_new(x) = poseN(...(pose2(pose1(x)))...)
        pose_new = pose_list[0]
        for pose in pose_list[1:]:
            pose_new = self.compose_pair(pose_new, pose)
        return pose_new

    def compose_pair(self, pose_a, pose_b):
        # pose_new(x) = pose_b(pose_a(x))
        R_a, t_a = pose_a[..., :3], pose_a[..., 3:]
        R_b, t_b = pose_b[..., :3], pose_b[..., 3:]
        R_new = R_b@R_a
        t_new = (R_b@t_a+t_b)[..., 0]
        pose_new = self(R=R_new, t=t_new)
        return pose_new

pose = Pose()

def compose_extrinsic_RT(RT: torch.Tensor):
    """
    Compose the standard form extrinsic matrix from RT.
    Batched I/O.
    """
    return torch.cat([
        RT,
        torch.tensor([[0, 0, 0, 1]], dtype=RT.dtype, device=RT.device)#.repeat(RT.shape[0], 1, 1)
        ], dim=0)

def get_ccm_pixel_grid(H, W):
    y_range = torch.arange(H, dtype=torch.float32)
    x_range = torch.arange(W, dtype=torch.float32)
    Y, X = torch.meshgrid(y_range, x_range, indexing='ij')
    Z = torch.ones_like(Y)
    xyz_grid = torch.stack([X, Y, Z],dim=-1).view(-1,3) 
    return xyz_grid

def ccm_to_main(opt, ccm, w2c, mask):
    H, W = opt.H, opt.W
    ccm[(mask <=0.5).view(-1, H*W)] = 0

    # proj to main cam, [B, 3, 3] @ [B, 3, H*W] -> B, H*W, 3
    seen_points = (w2c[:,:,:3]@ccm.permute(0,2,1)).permute(0,2,1).contiguous()
    #B, H*W, 3
    ori = w2c[:,:,3].unsqueeze(1).expand(seen_points.shape).contiguous()
    ori_radius = w2c[:,:,3].norm(dim=-1).unsqueeze(1)
    #B, H*W, 3
    seen_points = ori + seen_points
    seen_points[:,:,2] = seen_points[:,:,2] - ori_radius

    return seen_points

def vis_ccm_map(ccm, H, W, rgb=None, file_path=None):
    '''
    ccm: n_views, ch, H, W
    rgb: n_views, ch, H, W
    '''
    if len(ccm.shape) == 3:
        n_views = 1
        vis = ccm.squeeze(0)
    elif len(ccm.shape) == 4:
        n_views = ccm.shape[0]
    if file_path is None:
        file_path = f'test_{n_views}_views.png'
    # vis = ccm.reshape(n_views, H, W, 3).permute(0,3,1,2)
    vis = ccm
    # vis = (vis - vis.min()) / (vis.max() - vis.min())
    vis = (vis + 1) / 2
    if rgb is not None:
        if rgb.shape[-1] == vis.shape[-1]:
            vis = torch.cat([rgb, vis], dim=-2)
        else:
            img_fn = file_path.split('.')[0]
            img_fn = img_fn + '_rgb.png'
            rgb = rgb.squeeze(0).permute(1,2,0).cpu().numpy()
            Image.fromarray(rgb).save(img_fn)
            # save_image(rgb, img_fn)
    save_image(vis, file_path)
        

def depth_to_ccm(depth, intr, pose, mask):
    '''
    depth: [1, H, W]
    intr: [3, 3]
    pose: [4, 4]
    '''
    _, H, W = depth.shape
    depth = depth.squeeze(0)
    
    # [3, 3]
    K_inv = torch.linalg.inv(intr).float()
    # [3, 4]
    c2w = torch.linalg.inv(pose).float()[:3,:4]
    # [3, H*W]
    pixel_grid = get_ccm_pixel_grid(H, W).T
    # [3, H*W], in camera coordinates
    ray_dirs = K_inv @ pixel_grid.contiguous()
    # [H*W, 3] in world coordinates
    ray_dirs = (c2w[:,:3] @ ray_dirs).T
    #H*W, 3
    ray_oris = c2w[:,3].expand(ray_dirs.shape).contiguous() 
    # [H*W, 3], ccm
    seen_points = ray_oris + ray_dirs * depth.view(H*W, 1)
    # seen_points = (seen_points + 0.6) / 1.2
    seen_points[(mask <= 0.5).view(H*W)] = 0
    # [H, W, 3]
    seen_points = seen_points.reshape(H, W, 3)
    seen_points[...,1] *= -1 # Xinli accidently rotated, to match paper implementation

    # [3, H, W]
    return seen_points.permute(2, 0, 1)

def to_hom(X):
    '''
    X: [B, N, 3]
    Returns:
        X_hom: [B, N, 4]
    '''
    X_hom = torch.cat([X, torch.ones_like(X[..., :1])], dim=-1)
    return X_hom

def world2cam(X_world, pose):
    '''
    X_world: [B, N, 3]
    pose: [B, 3, 4]
    Returns:
        X_cam: [B, N, 3]
    '''
    X_hom = to_hom(X_world)
    X_cam = X_hom @ pose.transpose(-1, -2)
    return X_cam

def cam2img(X_cam, cam_intr):
    '''
    X_cam: [B, N, 3]
    cam_intr: [B, 3, 3]
    Returns:
        X_img: [B, N, 3]
    '''
    X_img = X_cam @ cam_intr.transpose(-1, -2)
    return X_img


######## Demo Pose Related #######

def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))

def circle_poses(radius=torch.tensor([4.6]), theta=torch.tensor([60]), phi=torch.tensor([0]), angle_overhead=30, angle_front=60):

    theta = theta / 180 * np.pi
    phi = phi / 180 * np.pi
    angle_overhead = angle_overhead / 180 * np.pi
    angle_front = angle_front / 180 * np.pi

    centers = torch.stack([
        radius * torch.sin(theta) * torch.sin(phi),
        radius * torch.sin(theta) * torch.cos(phi),
        radius * torch.cos(theta),
    ], dim=-1) # [B, 3]

    # lookat
    forward_vector = safe_normalize(centers)
    up_vector = torch.FloatTensor([0, 0, 1]).unsqueeze(0).repeat(len(centers), 1)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float).unsqueeze(0).repeat(len(centers), 1, 1)
    poses[:, :3, :3] = torch.stack((-right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    return poses.numpy()

def generate_circle_poses(radius=4.5, default_polar=90, size=8, render45=False):
    # luciddreamer code, ask yx if encounter issue
    rtn_poses = []
    thetas = torch.FloatTensor([default_polar*2//3]) if render45 else torch.FloatTensor([default_polar])
    radius = torch.FloatTensor([radius])
    for idx in range(size):
        phis = torch.FloatTensor([(idx / size) * 360])
        pose = circle_poses(radius, thetas, phis)
        rtn_poses.append(torch.tensor(pose))
    return torch.cat(rtn_poses)

def generate_lgm_poses(elevation=0, azimuth_max=360, radius=4.5):
    azimuth = np.arange(0, azimuth_max, 2, dtype=np.int32)
    rtn_poses = []
    for idx, azi in enumerate(tqdm(azimuth)):
        cam_poses = torch.from_numpy(orbit_camera(elevation, azi, radius=radius, opengl=True)).unsqueeze(0)
        rtn_poses.append(cam_poses)
    return torch.cat(rtn_poses)



def get_proj_matrix(znear=0.5, zfar=2.5,
                    fov=30.00000115370317):
    # assume fovx = fovy
    tan_half_fov = np.tan(0.5*np.deg2rad(fov))
    proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
    proj_matrix[0, 0] = 1 / tan_half_fov
    proj_matrix[1, 1] = 1 / tan_half_fov
    proj_matrix[2, 2] = (zfar + znear) / (zfar - znear)
    proj_matrix[3, 2] = - (zfar * znear) / (zfar - znear)
    proj_matrix[2, 3] = 1
    return proj_matrix

def convert_to_opencv(poses, proj_matrix):
    extr = poses[0]
    gs_cam_radius = extr[:,3].norm(dim=-1).unsqueeze(0)
    main_cam_transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, gs_cam_radius], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(extr)
    gs_cam_poses = main_cam_transform.unsqueeze(0) @ poses #[1,4,4] @ [V=8, 4, 4】
    gs_cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
    gs_cam_view = torch.inverse(gs_cam_poses).transpose(1, 2) #[V, 4, 4]
    gs_cam_view_proj = gs_cam_view @ proj_matrix
    gs_cam_pos = - gs_cam_poses[:, :3, 3]
    return gs_cam_view, gs_cam_view_proj, gs_cam_pos

######## Demo Pose Related #######