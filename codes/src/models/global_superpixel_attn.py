import torch
import torch.nn as nn



import torch.nn.functional as F
# from deepspeed.profiling.flops_profiler import get_model_profile
# from fvcore.nn import FlopCountAnalysis
# from fvcore.nn import flop_count_table






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

