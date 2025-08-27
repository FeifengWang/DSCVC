import numpy as np
import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from ptflops import get_model_complexity_info
from timm.layers import DropPath, trunc_normal_
from torch import nn


import torch.nn.functional as F

from src.layers.layers import conv3x3, conv1x1


def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )
class ResidualBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        if in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch)
        else:
            self.skip = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)

        if self.skip is not None:
            identity = self.skip(x)

        out = out + identity
        return out




class DGWA(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(DGWA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim // head_dim
        self.window_size = window_size
        self.type = type

        # 线性层用于偏移量预测
        self.offset_intra_layer = nn.Linear(self.input_dim, 2)
        self.offset_inter_layer1 = nn.Linear(self.input_dim, 2)
        self.offset_inter_layer2 = nn.Linear(self.window_size*self.window_size*2, 2)
        self.embedding_layer_q = nn.Linear(self.input_dim,  self.input_dim, bias=True)
        self.embedding_layer_kv = nn.Linear(self.input_dim, 2 * self.input_dim, bias=True)
        self.relative_position_params = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), self.n_heads))
        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=.02)
        self.relative_position_params = torch.nn.Parameter(
            self.relative_position_params.view(2 * window_size - 1, 2 * window_size - 1, self.n_heads).transpose(1,
                                                                                                                 2).transpose(
                0, 1))
    def generate_mask(self, h, w, p, shift):
        """ generating the mask of SW-MSA
        Args:
            shift: shift parameters in CyclicShift.
        Returns:
            attn_mask: should be (1 1 w p p),
        """
        attn_mask = torch.zeros(h, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.type == 'W':
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask
    def forward(self, x):
        if self.type != 'W':
            x = torch.roll(x, shifts=(-(self.window_size // 2), -(self.window_size // 2)), dims=(1, 2))



        # 在应用偏移量之前进行窗口划分
        x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        h_windows, w_windows = x.size(1), x.size(2)
        # 计算窗口内偏移
        offset_intra = self.offset_intra_layer(x)

        # 计算窗口间偏移
        offset_inter = self.offset_inter_layer1(x)
        offset_inter = self.offset_inter_layer2(rearrange(offset_inter, 'b w1 w2 p1 p2 c -> b w1 w2 (p1 p2 c)', p1=self.window_size, p2=self.window_size))

        # offset = rearrange(offset, 'b h w p1 p2 o -> b h w p1 p2 o', p1=self.window_size, p2=self.window_size, o=2)
        # 利用偏移量调整特征
        deform_x = self.apply_offsets(x, offset_intra, offset_inter, h_windows, w_windows)
        x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)

        deform_x = rearrange(deform_x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        q = self.embedding_layer_q(x)
        kv = self.embedding_layer_kv(deform_x)

        q = rearrange(q, 'b nw np (h c) -> h b nw np c', c=self.head_dim)
        k, v = rearrange(kv, 'b nw np (twoh c) -> twoh b nw np c', c=self.head_dim).chunk(2, dim=0)
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        sim += rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')

        if self.type != 'W':
            attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size // 2)
            sim = sim.masked_fill(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)

        if self.type != 'W':
            output = torch.roll(output, shifts=(self.window_size // 2, self.window_size // 2), dims=(1, 2))
        return output

    from einops import rearrange

    def apply_offsets(self, x, offset_intra, offset_inter, h_windows, w_windows):
        # 获取形状
        b, w1, w2, p1, p2, c = x.shape

        # 生成基准网格
        grid_y, grid_x = torch.meshgrid(torch.arange(w1), torch.arange(w2), indexing='ij')
        grid = torch.stack((grid_x, grid_y), -1).float().to(x.device)
        grid = grid.unsqueeze(0).unsqueeze(3).unsqueeze(3).expand(b, -1, -1, p1, p2, 2)

        # 应用窗口间偏移
        offset_grid_inter = grid + offset_inter.unsqueeze(3).unsqueeze(3).expand(-1, -1, -1, p1, p2, -1)

        # 归一化窗口间偏移
        offset_grid_inter = offset_grid_inter / ((max(w1, w2) - 1) / 2) - 1

        # 使用 rearrange 进行维度调整
        x = rearrange(x, 'b w1 w2 p1 p2 c -> (b p1 p2) c w1 w2')
        offset_grid_inter = rearrange(offset_grid_inter, 'b w1 w2 p1 p2 d -> (b p1 p2) w1 w2 d')

        x_deformed = F.grid_sample(x, offset_grid_inter, mode='bilinear', padding_mode='zeros', align_corners=True)

        # 恢复形状
        x_deformed = rearrange(x_deformed, '(b p1 p2) c w1 w2 -> b w1 w2 p1 p2 c', b=b, p1=p1, p2=p2)

        # 生成基准网格
        grid_y, grid_x = torch.meshgrid(torch.arange(p1), torch.arange(p2), indexing='ij')
        grid = torch.stack((grid_x, grid_y), -1).float().to(x.device)
        grid = grid.unsqueeze(0).unsqueeze(0).expand(b, w1 * w2, p1, p2, 2)

        # 应用窗口内偏移
        offset_grid_intra = grid + offset_intra.reshape(b, w1 * w2, p1, p2, 2)

        # 归一化窗口内偏移
        offset_grid_intra = offset_grid_intra / ((p1 - 1) / 2) - 1

        # 调整形状并在p1, p2上进行采样
        x_deformed = rearrange(x_deformed, 'b w1 w2 p1 p2 c -> (b w1 w2) c p1 p2')
        offset_grid_intra = rearrange(offset_grid_intra, 'b w1w2 p1 p2 d -> (b w1w2) p1 p2 d')

        x_deformed = F.grid_sample(x_deformed, offset_grid_intra, mode='bilinear', padding_mode='zeros',
                                   align_corners=True)

        # 恢复形状
        x_deformed = rearrange(x_deformed, '(b w1 w2) c p1 p2 -> b w1 w2 p1 p2 c', b=b, w1=w1, w2=w2)

        return x_deformed

    def relative_embedding(self):
        cord = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]))
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size - 1
        return self.relative_position_params[:, relation[:, :, 0].long(), relation[:, :, 1].long()]

class DGWABlock(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        """ SwinTransformer Block
        """
        super(DGWABlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ['W', 'SW']
        self.type = type
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = DGWA(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x):
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x




class IntraCtxBlock(nn.Module):
    def __init__(self, conv_dim, trans_dim, head_dim, window_size, drop_path, type='W'):
        """ SwinTransformer and Conv Block
        """
        super(IntraCtxBlock, self).__init__()
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        self.head_dim = head_dim
        self.window_size = window_size
        self.drop_path = drop_path
        self.type = type
        assert self.type in ['W', 'SW']
        self.trans_block = DGWABlock(self.trans_dim, self.trans_dim, self.head_dim, self.window_size, self.drop_path,
                                 self.type)
        self.conv1_1 = nn.Conv2d(self.conv_dim + self.trans_dim, self.conv_dim + self.trans_dim, 1, 1, 0, bias=True)
        self.conv1_2 = nn.Conv2d(self.conv_dim + self.trans_dim, self.conv_dim + self.trans_dim, 1, 1, 0, bias=True)

        self.conv_block = ResidualBlock(self.conv_dim + self.trans_dim, self.conv_dim)

    def forward(self, x,g):
        x_intra, _ = x.chunk(2, 1)
        # conv_x, trans_x = torch.split(self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1)
        conv_x = self.conv_block(x)
        ctx_intra = Rearrange('b c h w -> b h w c')(x_intra)
        ctx_intra = self.trans_block(ctx_intra)
        ctx_intra = Rearrange('b h w c -> b c h w')(ctx_intra)
        ctx_intra = ctx_intra*(1-g)
        res = self.conv1_2(torch.cat((conv_x, ctx_intra), dim=1))
        x = x + res
        return x

if __name__ == '__main__':
    x = torch.ones((2,64,32,32))
    ref = torch.ones((2,64,32,32))

    # net = DeformableConvTransBlock(64,64,8,8,0)
    net = DGWABlock(64,64,8,8,0)
    # y = net(torch.cat((x,ref),dim=1))
    # print(y.shape)
    print("Total number of parameters in network is {}".format(sum(x.numel() for x in net.parameters())))
    # print("Total cnn of parameters in network is {}".format(sum(x.numel() for x in net.trans_block.msa.feature_extractor.parameters())))
    # from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(net, (64, 64, 64), as_strings=True, verbose=True)
    print('{:<30}  {:<8}'.format('FLOPs: ', macs))
    print('{:<30}  {:<8}'.format('Parameters: ', params))
    print("ok")