import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.util import EasyDict as edict
from utils.gs_loss import Loss
from diffusers import AutoencoderKL
from utils.util import toggle_grad, get_child_state_dict
from models.gs_core.gs import GaussianRenderer
from einops import rearrange
# from torchvision.utils import save_image
# torch.autograd.set_detect_anomaly(True)

class Graph(nn.Module):
    def __init__(self, opt):
        super().__init__()
        if not opt.arch.svd:
            from models.gs_core.stable_diffusion import StableDiffusion
            if opt.model_key:
                vae_model_key = '/hpc2hdd/home/hheat/projects/shape_ccm/pretrained/stable-diffusion-2-1-base/vae'
            else:
                vae_model_key = 'stabilityai/stable-diffusion-2-1/vae'
        else:
            from models.gs_core.stable_video_diffusion import StableDiffusion
            if opt.model_key:
                vae_model_key = '/hpc2hdd/home/hheat/projects/shape_ccm/pretrained/zeroscope_v2_576w/vae'
            else:
                vae_model_key = 'cerspense/zeroscope_v2_576w/vae'

        self.dpt_depth = StableDiffusion(opt, get_gs_feat=True)
        if opt.pretrain.depth and not opt.resume:
            self.load_pretrained_depth(opt)
        if opt.optim.fix_dpt:
            toggle_grad(self.dpt_depth, False)


        vae = AutoencoderKL.from_pretrained(vae_model_key, torchdtype=torch.float32, 
                                            latent_channels=320, out_channels=11, low_cpu_mem_usage=False, 
                                            ignore_mismatched_sizes=True)
        vae.enable_xformers_memory_efficient_attention()
        self.gs_decoder = vae.decoder
        self.pos_act = lambda x: torch.tanh(x)
        self.scale_act = lambda x: 0.1 * F.softplus(x)
        self.opacity_act = lambda x: torch.sigmoid(x)
        self.rot_act = lambda x: F.normalize(x, dim=-1)
        self.rgb_act = lambda x: torch.tanh(x)

        self.gs = GaussianRenderer(image_height=opt.H, image_width=opt.W, zfar=opt.rendering.zfar, znear=opt.rendering.znear, fovy=opt.fovy)
        self.loss_fns = Loss(opt)

    def load_pretrained_depth(self, opt):
        # loading from our pretrained depth and intr model
        if opt.device == 0:
            print("loading dpt depth from {}...".format(opt.pretrain.depth))
        checkpoint = torch.load(opt.pretrain.depth, map_location="cuda:{}".format(opt.device))
        self.dpt_depth.load_state_dict(get_child_state_dict(checkpoint["graph"], "dpt_depth"))
        

    def forward(self, opt, var, training=False, get_loss=True, ccm_only=False):
        batch_size = len(var.idx)
        slicer = opt.input_frames if training else opt.test_input_frames
        if training:
            if opt.random_num_input:
                # num input frame default is 5, so we pick from 1 to 5
                num_slice = torch.randint(low=1, high=opt.input_frames+1, size=(1,)).item() # 1 to 5 views
                if opt.arch.svd:
                    var.rgb_input_map = var.rgb_input_map[:, 0:num_slice, :, :, :]
                    var.ref_mask_input_map = var.ref_mask_input_map[:, 0:num_slice, :, :, :]
                else:
                    var.rgb_input_map = var.rgb_input_map[:, :, :, 0:num_slice*var.rgb_input_map.shape[-2]] #B, 3, H, W*V -> B, 3, H, W*num_slice
                    var.ref_mask_input_map = var.ref_mask_input_map[:, 0:num_slice, :, :, :] # B, V, 3, H, W -> B, num_slice, 3, H, W

        seen_points_3D_pred, gs_feat = self.dpt_depth(var.rgb_input_map) # [B, V, 3, H, W], [(B V), 320, H/8, W/8]
        _f = var.ref_mask_input_map.shape[1]
        if not opt.arch.svd:
            seen_points_3D_pred = torch.cat(torch.chunk(seen_points_3D_pred, _f, -1), dim=1) #B, V, 3, H, W
            gs_feat = torch.cat(torch.chunk(gs_feat, _f, -1), dim=1) #B, V, 320, H/8, W/8
            var.rgb_input_map = torch.stack(torch.chunk(var.rgb_input_map, _f, -1), dim=1) #B, V, 3, H, W
        batch_size, n_views, ch, h, w = seen_points_3D_pred.shape

        gs_feat = rearrange(gs_feat, "b v c h w -> (b v) c h w")

        seen_points_3D_pred = rearrange(seen_points_3D_pred, "b v c h w -> b (v h w) c")   

        if opt.use_mask:
            seen_points_3D_pred[:,:slicer*(var.rgb_input_map.shape[-1] * var.rgb_input_map.shape[-2]),:][(var.ref_mask_input_map<=0.5).view(batch_size, -1)] = 0
        var.seen_points_pred = seen_points_3D_pred[:, :slicer*(var.rgb_input_map.shape[-1] * var.rgb_input_map.shape[-2]), :]
        var.seen_points_pred = rearrange(var.seen_points_pred, 'b (v h w) c -> (b v) (h w) c', v=_f, h=var.rgb_input_map.shape[-2])

        if ccm_only:
            return var

        ######## gs rendering #######
        x = self.gs_decoder(gs_feat) 
        x = rearrange(x, '(b v) c h w -> b (v h w) c', v=n_views)
        pos = seen_points_3D_pred # B, N, 3
        pos[:,:,2] = pos[:,:,2] * -1
        opacity = self.opacity_act(x[..., 0:1])
        scale = self.scale_act(x[..., 1:4])
        rotation = self.rot_act(x[..., 4:8])
        rgbs = self.rgb_act(x[..., 8:])

        # if not get_loss:
        #     var.opacity = opacity.clone().detach() # for vis opacity [B, V*H*W, 1]
        #     var.opacity = (var.opacity>0.5).float()
        #     var.opacity[:,:slicer*(var.rgb_input_map.shape[-1] * var.rgb_input_map.shape[-2]),:][(var.ref_mask_input_map<=0.5).view(batch_size, -1)] = 0

        rgb_raw = rearrange(var.rgb_input_map, 'b v c h w -> b (v h w) c')
        rgbs = rgbs + rgb_raw

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1) # [B, N, 14]

        bg_color = torch.ones(3, dtype=torch.float32, device=gaussians.device)
        if 'gs_images_gt' in var:
            var.gs_images_gt = var.gs_images_gt * var.gs_masks_gt + bg_color.view(1, 1, 3, 1, 1) * (1 - var.gs_masks_gt)
        results = self.gs.render(gaussians, var.gs_cam_view, var.gs_cam_view_proj, var.gs_cam_pos, bg_color=bg_color)
        var.pred_images = results['image'] # [B, V, C, output_size, output_size]
        var.pred_alphas = results['alpha'] # [B, V, 1, output_size, output_size]
            
        # calculate the loss if needed
        if get_loss: 
            loss = self.compute_loss(opt, var, training)
            return var, loss
        return var

    def compute_loss(self, opt, var, training=False):
        loss = edict()
        if opt.loss_weight.gs_mse is not None:
            H,W = var.pred_images.shape[-2:]
            dssim = 0.2 #test 暂时对齐gs, clean later
            loss.gs_mse = (1 - dssim) * ( self.loss_fns.gs_mse_loss(var.pred_images, var.gs_images_gt) + \
                            self.loss_fns.gs_mse_loss(var.pred_alphas, var.gs_masks_gt)) + \
                            dssim * (1 - self.loss_fns.ssim(var.pred_images.view(-1, 3, H, W), var.gs_images_gt.view(-1, 3, H, W)))
        if opt.loss_weight.gs_lpips is not None:
            H,W = var.pred_images.shape[-2:]
            loss.gs_lpips = self.loss_fns.gs_lpips_loss(
                F.interpolate(var.pred_images.view(-1, 3, H, W) * 2 - 1, (256, 256), mode='bilinear', align_corners=False), 
                F.interpolate(var.gs_images_gt.view(-1, 3, H, W) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
            ).mean()
        return loss

