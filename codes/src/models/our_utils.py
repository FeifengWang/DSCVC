from typing import Optional, Union, List, Any, Tuple

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import warnings

from compressai.entropy_models import EntropyModel
from compressai.layers.layers import conv1x1
from compressai.ops import LowerBound
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from torchvision import transforms

from compressai.layers import *
from spatial_correlation_sampler import SpatialCorrelationSampler as Correlation


def tensor_to_PIL(tensor):
    unloader = transforms.ToPILImage()

    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image
def pic2cu(tensor, crop_size):
    B, C, H, W = tensor.shape
    num_in_col = H // crop_size
    num_in_row = W // crop_size
    patchlist = []
    for i in range(num_in_col):
        for j in range(num_in_row):
            patch = tensor[:, :, i * crop_size:(i + 1) * crop_size, j * crop_size:(j + 1) * crop_size]
            patchlist.append(patch)
    list_tensor = torch.cat(patchlist,dim=1)
    # list_tensor = torch.stack(patchlist) #stack + permute + view/reshape =cat
    # list_tensor = list_tensor.permute(1, 0, 2, 3, 4).contiguous()#必须是10234 不能是12034
    # list_tensor = list_tensor.reshape((B, -1, crop_size, crop_size))
    return list_tensor

def cu2pic(patch_tensor,pic_tensor,crop_size,nc):
    B, C, H, W = pic_tensor.shape
    num_in_col = H // crop_size
    num_in_row = W // crop_size
    idx = 0
    for i in range(num_in_col):
        for j in range(num_in_row):
            pic_tensor[:, :, i * crop_size:(i + 1) * crop_size, j * crop_size:(j + 1) * crop_size] = patch_tensor[:, nc * idx:nc * (idx + 1), :, :]
            idx +=1
    return pic_tensor


class Conv_relu(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size, stride, padding, has_relu=True, efficient=False):
        super(Conv_relu, self).__init__()
        self.has_relu = has_relu
        self.efficient = efficient

        self.conv = nn.Conv2d(in_chl, out_chl, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        def _func_factory(conv, relu, has_relu):
            def func(x):
                x = conv(x)
                if has_relu:
                    x = relu(x)
                return x

            return func

        func = _func_factory(self.conv, self.relu, self.has_relu)

        if self.efficient:
            x = checkpoint(func, x)
        else:
            x = func(x)

        return x


class Aggregate(nn.Module):
    def __init__(self, nf=64, nbr=3, n_group=8, kernels=[3, 3, 3, 3], patches=[7, 11, 15], cor_ksize=3):
        super(Aggregate, self).__init__()
        self.nbr = nbr
        self.cas_k = kernels[0]
        self.k1 = kernels[1]
        self.k2 = kernels[2]
        self.k3 = kernels[3]
        self.g = n_group

        self.L3_conv1 = Conv_relu(nf * 2, nf, 3, 1, 1, has_relu=True)
        self.L3_conv2 = Conv_relu(nf, nf, 3, 1, 1, has_relu=True)
        self.L3_conv3 = Conv_relu(nf, nf, (7, 1), 1, (3, 0), has_relu=True)
        self.L3_conv4 = Conv_relu(nf, nf, (1, 7), 1, (0, 3), has_relu=True)
        self.L3_mask = Conv_relu(nf, self.g * self.k3 ** 2, self.k3, 1, (self.k3 - 1) // 2, has_relu=False)
        self.L3_nn_conv = Conv_relu(nf * self.nbr, nf, 3, 1, 1, has_relu=True)

        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.patch_size = patches
        self.cor_k = cor_ksize
        self.padding = (self.cor_k - 1) // 2
        self.pad_size = [self.padding + (patch - 1) // 2 for patch in self.patch_size]
        self.add_num = [2 * pad - self.cor_k + 1 for pad in self.pad_size]
        self.L3_corr = Correlation(kernel_size=self.cor_k, patch_size=self.patch_size[0],
                                   stride=1, padding=self.padding, dilation=1, dilation_patch=1)
        self.final_conv = Conv_relu(nf * 2, nf, 3, 1, 1, has_relu=True)

    def forward(self, ref_fea_l):
        # L3
        nbr_fea_l = ref_fea_l
        B, C, H, W = nbr_fea_l.size()
        L3_w = torch.cat([nbr_fea_l, ref_fea_l], dim=1)
        L3_w = self.L3_conv4(self.L3_conv3(self.L3_conv2(self.L3_conv1(L3_w))))
        L3_mask = self.L3_mask(L3_w).view(B, self.g, 1, self.k3 ** 2, H, W)
        # corr: B, (2 * dis + 1) ** 2, H, W
        L3_norm_ref_fea = F.normalize(ref_fea_l, dim=1)
        L3_norm_nbr_fea = F.normalize(nbr_fea_l, dim=1)
        L3_corr = self.L3_corr(L3_norm_ref_fea, L3_norm_nbr_fea).view(B, -1, H, W)
        # corr_ind: B, H, W
        _, L3_corr_ind = torch.topk(L3_corr, self.nbr, dim=1)
        L3_corr_ind = L3_corr_ind.permute(0, 2, 3, 1).reshape(B, H * W * self.nbr)
        L3_ind_row_add = L3_corr_ind // self.patch_size[0] * (W + self.add_num[0])
        L3_ind_col_add = L3_corr_ind % self.patch_size[0]
        L3_corr_ind = L3_ind_row_add + L3_ind_col_add
        # generate top-left indexes
        y = torch.arange(H).repeat_interleave(W).to(ref_fea_l)
        x = torch.arange(W).repeat(H).to(ref_fea_l)
        L3_lt_ind = y * (W + self.add_num[0]) + x
        L3_lt_ind = L3_lt_ind.repeat_interleave(self.nbr).long().unsqueeze(0)
        L3_corr_ind = (L3_corr_ind + L3_lt_ind).view(-1)
        # L3_nbr: B, 64 * k * k, (H + 2 * pad - k + 1) * (W + 2 * pad -k + 1)
        L3_nbr = F.unfold(nbr_fea_l, self.cor_k, dilation=1, padding=self.pad_size[0], stride=1)
        ind_B = torch.arange(B, dtype=torch.long).repeat_interleave(H * W * self.nbr).to(ref_fea_l)
        # L3: B * H * W * nbr, 64 * k * k
        L3 = L3_nbr[ind_B.long(), :, L3_corr_ind].view(B * H * W, self.nbr * C, self.cor_k, self.cor_k)
        L3 = self.L3_nn_conv(L3)
        L3 = L3.view(B, H, W, C, self.cor_k ** 2).permute(0, 3, 4, 1, 2)
        L3 = L3.view(B, self.g, C // self.g, self.cor_k ** 2, H, W)
        L3 = self.relu((L3 * L3_mask).sum(dim=3).view(B, C, H, W))
        L3 = self.final_conv(torch.cat((ref_fea_l,L3),dim=1))
        return L3


class AggregateAttentionBlock(nn.Module):
    """Self attention block.

    Simplified variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Args:
        N (int): Number of channels)
    """

    def __init__(self, N: int, reverse: bool):
        super().__init__()
        self.reverse = reverse

        class ResidualUnit(nn.Module):
            """Simple residual unit."""

            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    conv1x1(N, N // 2),
                    nn.ReLU(inplace=True),
                    conv3x3(N // 2, N // 2),
                    nn.ReLU(inplace=True),
                    conv1x1(N // 2, N),
                )
                self.relu = nn.ReLU(inplace=True)

            def forward(self, x: Tensor) -> Tensor:
                identity = x
                out = self.conv(x)
                out += identity
                out = self.relu(out)
                return out

        self.aggr = nn.Sequential(conv1x1(N, 64), Aggregate(nf=64, nbr=3, n_group=8), conv1x1(64, N))

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        # a = self.conv_a(x)
        # b = self.conv_b(x)
        out = self.aggr(x)
        if self.reverse:
            out = identity + out
        else:
            out = identity + out
        return out