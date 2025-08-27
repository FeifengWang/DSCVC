import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.layers import ResidualBlock
# --------------------------------------------------------
# Super Token Vision Transformer (STViT)
# Copyright (c) 2023 CASIA
# Licensed under The MIT License [see LICENSE for details]
# Written by Huaibo Huang
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from deepspeed.profiling.flops_profiler import get_model_profile
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import scipy.io as sio
import torch.nn.functional as F
import math
from functools import partial
# from fvcore.nn import FlopCountAnalysis
# from fvcore.nn import flop_count_table
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import time


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()


class ResDWC(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()

        self.dim = dim
        self.kernel_size = kernel_size

        self.conv = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size // 2, groups=dim)

        # self.conv_constant = nn.Parameter(torch.eye(kernel_size).reshape(dim, 1, kernel_size, kernel_size))
        # self.conv_constant.requires_grad = False

    def forward(self, x):
        # return F.conv2d(x, self.conv.weight+self.conv_constant, self.conv.bias, stride=1, padding=self.kernel_size//2, groups=self.dim) # equal to x + conv(x)
        return x + self.conv(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., conv_pos=True,
                 downsample=False, kernel_size=5):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features = out_features or in_features
        self.hidden_features = hidden_features = hidden_features or in_features

        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act1 = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

        self.conv = ResDWC(hidden_features, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop(x)
        x = self.conv(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, window_size=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.window_size = window_size

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        q, k, v = self.qkv(x).reshape(B, self.num_heads, C // self.num_heads * 3, N).chunk(3,
                                                                                           dim=2)  # (B, num_heads, head_dim, N)

        attn = (k.transpose(-1, -2) @ q) * self.scale

        attn = attn.softmax(dim=-2)  # (B, h, N, N)
        attn = self.attn_drop(attn)

        x = (v @ attn).reshape(B, C, H, W)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Unfold(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()

        self.kernel_size = kernel_size

        weights = torch.eye(kernel_size ** 2)
        weights = weights.reshape(kernel_size ** 2, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)

    def forward(self, x):
        b, c, h, w = x.shape
        x = F.conv2d(x.reshape(b * c, 1, h, w), self.weights, stride=1, padding=self.kernel_size // 2)
        return x.reshape(b, c * 9, h * w)


class Fold(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()

        self.kernel_size = kernel_size

        weights = torch.eye(kernel_size ** 2)
        weights = weights.reshape(kernel_size ** 2, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)

    def forward(self, x):
        b, _, h, w = x.shape
        x = F.conv_transpose2d(x, self.weights, stride=1, padding=self.kernel_size // 2)
        return x



class GobalSuperPixelAttention(nn.Module):
    def __init__(self, dim, superpixel_size=[8,8], n_iter=1, refine=True, refine_attention=True, num_heads=4, qkv_bias=True,
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.n_iter = n_iter
        self.superpixel_size = superpixel_size
        self.refine = refine
        self.refine_attention = refine_attention

        self.scale = dim ** - 0.5

        self.unfold = Unfold(3)
        self.fold = Fold(3)
        self.kmeans_iters = 1  # Number of KMeans iterations for initialization

        # Learnable parameters for KMeans initialization
        self.kmeans_centroids = nn.Parameter(torch.randn(1, dim, 1, 1))
        if refine:

            if refine_attention:
                self.stoken_refine = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                               attn_drop=attn_drop, proj_drop=proj_drop)
            else:
                self.stoken_refine = nn.Sequential(
                    nn.Conv2d(dim, dim, 1, 1, 0),
                    nn.Conv2d(dim, dim, 5, 1, 2, groups=dim),
                    nn.Conv2d(dim, dim, 1, 1, 0)
                )
    def kmeans_init(self, x, h, w):
        """
        Differentiable KMeans-like initialization for superpixels

        Args:
            x: Input features (B, C, H, W)
            h, w: Superpixel grid dimensions
            hh, ww: Superpixel dimensions (H/h, W/w)

        Returns:
            Initial superpixel centroids (B, C, h, w)
        """
        B, C, H, W = x.shape

        # Initialize centroids with grid sampling
        centroids = F.adaptive_avg_pool2d(x, (h, w))

        # Add learnable perturbation to initial centroids
        # centroids = centroids + self.kmeans_centroids

        # Run a few iterations of differentiable KMeans
        for _ in range(self.kmeans_iters):
            # Reshape pixels and centroids for distance computation
            pixels = x.reshape(B, C, H * W).permute(0, 2, 1)  # B, HW, C
            centroids_flat = centroids.reshape(B, C, h * w).permute(0, 2, 1)  # B, hw, C

            # Compute distances between pixels and centroids
            distances = torch.cdist(pixels, centroids_flat, p=2)  # B, HW, hw

            # Create soft assignments (differentiable version of hard assignment)
            assignments = F.softmax(-distances, dim=-1)  # B, HW, hw

            # Update centroids with weighted average
            weights = assignments.permute(0, 2, 1)  # B, hw, HW
            weighted_pixels = torch.bmm(weights, pixels)  # B, hw, C
            sum_weights = weights.sum(dim=-1, keepdim=True) + 1e-6
            new_centroids = weighted_pixels / sum_weights

            # Reshape back to spatial dimensions
            centroids = new_centroids.permute(0, 2, 1).reshape(B, C, h, w)

        return centroids


    def forward(self, x_cur, x_ref):
        '''
        Compute global superpixel attention between two inputs
        x_cur: (B, C, H, W) - current frame features
        x_ref: (B, C, H, W) - reference frame features
        '''
        B, C, H0, W0 = x_cur.shape
        h, w = self.superpixel_size

        pad_l = pad_t = 0
        pad_r = (w - W0 % w) % w
        pad_b = (h - H0 % h) % h
        if pad_r > 0 or pad_b > 0:
            x_cur = F.pad(x_cur, (pad_l, pad_r, pad_t, pad_b))
            x_ref = F.pad(x_ref, (pad_l, pad_r, pad_t, pad_b))

        _, _, H, W = x_cur.shape
        hh, ww = H // h, W // w

        # Initialize superpixel features using reference frame
        superpixel_features = self.kmeans_init(x_ref, hh, ww)

        # Reshape current frame features for attention computation
        pixel_features_cur = x_cur.reshape(B, C, hh, h, ww, w).permute(0, 2, 4, 3, 5, 1).reshape(B, hh * ww, h * w, C)

        with torch.no_grad():
            for idx in range(self.n_iter):
                superpixel_features = self.unfold(superpixel_features)  # (B, C*9, hh*ww)
                superpixel_features = superpixel_features.transpose(1, 2).reshape(B, hh * ww, C, 9)
                affinity_matrix = pixel_features_cur @ superpixel_features * self.scale  # (B, hh*ww, h*w, 9)
                affinity_matrix = affinity_matrix.softmax(-1)  # (B, hh*ww, h*w, 9)
                affinity_matrix_sum = affinity_matrix.sum(2).transpose(1, 2).reshape(B, 9, hh, ww)
                affinity_matrix_sum = self.fold(affinity_matrix_sum)
                if idx < self.n_iter - 1:
                    # Assign the pixel to update superpixel features using current frame
                    superpixel_features = pixel_features_cur.transpose(-1, -2) @ affinity_matrix  # (B, hh*ww, C, 9)
                    superpixel_features = self.fold(superpixel_features.permute(0, 2, 3, 1).reshape(B * C, 9, hh, ww)).reshape(
                        B, C, hh, ww)
                    superpixel_features = superpixel_features / (affinity_matrix_sum + 1e-12)  # (B, C, hh, ww)

        # Final superpixel features using current frame
        superpixel_features = pixel_features_cur.transpose(-1, -2) @ affinity_matrix  # (B, hh*ww, C, 9)
        superpixel_features = self.fold(superpixel_features.permute(0, 2, 3, 1).reshape(B * C, 9, hh, ww)).reshape(B, C, hh, ww)
        superpixel_features = superpixel_features / (affinity_matrix_sum.detach() + 1e-12)  # (B, C, hh, ww)

        if self.refine:
            if self.refine_attention:
                superpixel_features = self.stoken_refine(superpixel_features)
            else:
                superpixel_features = self.stoken_refine(superpixel_features)

        # Apply attention to reference frame features
        superpixel_features = self.unfold(superpixel_features)  # (B, C*9, hh*ww)
        superpixel_features = superpixel_features.transpose(1, 2).reshape(B, hh * ww, C, 9)  # (B, hh*ww, C, 9)
        pixel_features = superpixel_features @ affinity_matrix.transpose(-1, -2)  # (B, hh*ww, C, h*w)
        pixel_features = pixel_features.reshape(B, hh, ww, C, h, w).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)

        if pad_r > 0 or pad_b > 0:
            pixel_features = pixel_features[:, :, :H0, :W0]

        return pixel_features





class SuperPixelAttention(nn.Module):
    def __init__(self, dim, superpixel_size=[8,8], n_iter=1, refine=True, refine_attention=True, num_heads=4, qkv_bias=True,
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.n_iter = n_iter
        self.superpixel_size = superpixel_size
        self.refine = refine
        self.refine_attention = refine_attention

        self.scale = dim ** - 0.5

        self.unfold = Unfold(3)
        self.fold = Fold(3)
        self.kmeans_iters = 1  # Number of KMeans iterations for initialization

        # Learnable parameters for KMeans initialization
        self.kmeans_centroids = nn.Parameter(torch.randn(1, dim, 1, 1))
        if refine:

            if refine_attention:
                self.stoken_refine = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                               attn_drop=attn_drop, proj_drop=proj_drop)
            else:
                self.stoken_refine = nn.Sequential(
                    nn.Conv2d(dim, dim, 1, 1, 0),
                    nn.Conv2d(dim, dim, 5, 1, 2, groups=dim),
                    nn.Conv2d(dim, dim, 1, 1, 0)
                )
    def kmeans_init(self, x, h, w):
        """
        Differentiable KMeans-like initialization for superpixels

        Args:
            x: Input features (B, C, H, W)
            h, w: Superpixel grid dimensions
            hh, ww: Superpixel dimensions (H/h, W/w)

        Returns:
            Initial superpixel centroids (B, C, h, w)
        """
        B, C, H, W = x.shape

        # Initialize centroids with grid sampling
        centroids = F.adaptive_avg_pool2d(x, (h, w))

        # Add learnable perturbation to initial centroids
        # centroids = centroids + self.kmeans_centroids

        # Run a few iterations of differentiable KMeans
        for _ in range(self.kmeans_iters):
            # Reshape pixels and centroids for distance computation
            pixels = x.reshape(B, C, H * W).permute(0, 2, 1)  # B, HW, C
            centroids_flat = centroids.reshape(B, C, h * w).permute(0, 2, 1)  # B, hw, C

            # Compute distances between pixels and centroids
            distances = torch.cdist(pixels, centroids_flat, p=2)  # B, HW, hw

            # Create soft assignments (differentiable version of hard assignment)
            assignments = F.softmax(-distances, dim=-1)  # B, HW, hw

            # Update centroids with weighted average
            weights = assignments.permute(0, 2, 1)  # B, hw, HW
            weighted_pixels = torch.bmm(weights, pixels)  # B, hw, C
            sum_weights = weights.sum(dim=-1, keepdim=True) + 1e-6
            new_centroids = weighted_pixels / sum_weights

            # Reshape back to spatial dimensions
            centroids = new_centroids.permute(0, 2, 1).reshape(B, C, h, w)

        return centroids


    def cross_superpixel_forward(self, x_cur, x_ref):
        '''
        Compute cross superpixel attention between two inputs
        x_cur: (B, C, H, W) - current frame features
        x_ref: (B, C, H, W) - reference frame features
        '''
        B, C, H0, W0 = x_cur.shape
        h, w = self.superpixel_size

        pad_l = pad_t = 0
        pad_r = (w - W0 % w) % w
        pad_b = (h - H0 % h) % h
        if pad_r > 0 or pad_b > 0:
            x_cur = F.pad(x_cur, (pad_l, pad_r, pad_t, pad_b))
            x_ref = F.pad(x_ref, (pad_l, pad_r, pad_t, pad_b))

        _, _, H, W = x_cur.shape
        hh, ww = H // h, W // w

        # Initialize superpixel features using reference frame
        superpixel_features = self.kmeans_init(x_ref, hh, ww)

        # Reshape current frame features for attention computation
        pixel_features_cur = x_cur.reshape(B, C, hh, h, ww, w).permute(0, 2, 4, 3, 5, 1).reshape(B, hh * ww, h * w, C)

        with torch.no_grad():
            for idx in range(self.n_iter):
                superpixel_features = self.unfold(superpixel_features)  # (B, C*9, hh*ww)
                superpixel_features = superpixel_features.transpose(1, 2).reshape(B, hh * ww, C, 9)

                # pixel assignments
                affinity_matrix = pixel_features_cur @ superpixel_features * self.scale  # (B, hh*ww, h*w, 9)
                affinity_matrix = affinity_matrix.softmax(-1)  # (B, hh*ww, h*w, 9)
                affinity_matrix_sum = affinity_matrix.sum(2).transpose(1, 2).reshape(B, 9, hh, ww)
                affinity_matrix_sum = self.fold(affinity_matrix_sum)
                if idx < self.n_iter - 1:
                    # Update superpixel features using current frame
                    superpixel_features = pixel_features_cur.transpose(-1, -2) @ affinity_matrix  # (B, hh*ww, C, 9)
                    superpixel_features = self.fold(superpixel_features.permute(0, 2, 3, 1).reshape(B * C, 9, hh, ww)).reshape(
                        B, C, hh, ww)
                    superpixel_features = superpixel_features / (affinity_matrix_sum + 1e-12)  # (B, C, hh, ww)

        # Final superpixel features using current frame
        superpixel_features = pixel_features_cur.transpose(-1, -2) @ affinity_matrix  # (B, hh*ww, C, 9)
        superpixel_features = self.fold(superpixel_features.permute(0, 2, 3, 1).reshape(B * C, 9, hh, ww)).reshape(B, C, hh, ww)
        superpixel_features = superpixel_features / (affinity_matrix_sum.detach() + 1e-12)  # (B, C, hh, ww)

        if self.refine:
            if self.refine_attention:
                superpixel_features = self.stoken_refine(superpixel_features)
            else:
                superpixel_features = self.stoken_refine(superpixel_features)

        # Apply attention to reference frame features
        superpixel_features = self.unfold(superpixel_features)  # (B, C*9, hh*ww)
        superpixel_features = superpixel_features.transpose(1, 2).reshape(B, hh * ww, C, 9)  # (B, hh*ww, C, 9)
        pixel_features = superpixel_features @ affinity_matrix.transpose(-1, -2)  # (B, hh*ww, C, h*w)
        pixel_features = pixel_features.reshape(B, hh, ww, C, h, w).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)

        if pad_r > 0 or pad_b > 0:
            pixel_features = pixel_features[:, :, :H0, :W0]

        return pixel_features

    def direct_forward(self, x):
        B, C, H, W = x.shape
        superpixel_features = x
        if self.refine:
            if self.refine_attention:
                # superpixel_features = superpixel_features.flatten(2).transpose(-1, -2)
                superpixel_features = self.stoken_refine(superpixel_features)
                # superpixel_features = superpixel_features.transpose(-1, -2).reshape(B, C, H, W)
            else:
                superpixel_features = self.stoken_refine(superpixel_features)
        return superpixel_features

    def forward(self,x_cur):
        x_ref = x_cur
        return self.cross_superpixel_forward(x_cur, x_ref)

def analyze_flops_macs(superpixel_size):
    # 创建模型实例
    model = SuperPixelAttention(dim=64, superpixel_size=superpixel_size)

    img_shape = (1,64,64,64)




    flops, macs, params = get_model_profile(model, img_shape,as_string=False)


    print("flops: ", flops)
    print("macs: ", macs)

    return {
        'superpixel_size': superpixel_size,
        'total_flops': flops,
        'total_macs': macs,
        'total_params': params
    }

if __name__ == '__main__':
    # 分析不同superpixel_size值下的FLOPs和MACs
    superpixel_sizes = [[1, 1], [4, 4], [8, 8], [16, 16]]
    results = []

    print("CrossSuperPixelAttention FLOPs/MACs分析:")
    print("="*60)

    for size in superpixel_sizes:
        try:
            result = analyze_flops_macs(size)
            results.append(result)

            print(f"\nsuperpixel_size={size}:")
            print(f"  参数量: {result['total_params']:,}")
            print(f"  总FLOPs: {result['total_flops']:,}")
            print(f"  总FLOPs (GFLOPs): {result['total_flops'] / 1e9:.4f} GFLOPs")
            # MACs通常约等于FLOPs的一半
            print(f"  估计MACs (GMACs): {result['total_macs'] / 1e9:.4f} GMACs")

        except Exception as e:
            print(f"superpixel_size={size} 分析失败: {e}")

    print("\n" + "="*60)

