import torch
import torch.nn as nn
import torch.nn.functional as torch_F

from copy import deepcopy
from kiui.lpips import LPIPS
from math import exp
from torch.autograd import Variable

class Loss(nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.opt = deepcopy(opt)
        self.mse_loss = nn.MSELoss()
        self.lpips_loss = LPIPS(net='vgg')
        self.lpips_loss.requires_grad_(False)
    
    def gs_mse_loss(self, pred, gt):
        assert len(pred.shape) == len(gt.shape) == 5
        return self.mse_loss(pred, gt)

    def gs_lpips_loss(self, pred, gt):
        assert len(pred.shape) == len(gt.shape) == 4
        return self.lpips_loss(pred, gt)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    def ssim(self,img1, img2, window_size=11, size_average=True):
        channel = img1.size(-3)
        window = self.create_window(window_size, channel)

        if img1.is_cuda:
            window = window.cuda(img1.get_device())
        window = window.type_as(img1)

        return self._ssim(img1, img2, window, window_size, channel, size_average)

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = torch_F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = torch_F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = torch_F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = torch_F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = torch_F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)