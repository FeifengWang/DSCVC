import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function

from .global_superpixel_attn import GobalSuperPixelAttention

from ..layers.layers import subpel_conv1x1, conv3x3, \
    ResidualBlock, ResidualBlockWithStride, ResidualBlockUpsample
import torchvision.ops.deform_conv as df

backward_grid = [{} for _ in range(9)]  # 0~7 for GPU, -1 for CPU


# pylint: disable=W0221
class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones_like(inputs) * bound
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors
        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


# pylint: enable=W0221


def torch_warp(feature, flow, mode='bilinear'):
    device_id = -1 if feature.device == torch.device('cpu') else feature.device.index
    if str(flow.size()) not in backward_grid[device_id]:
        N, _, H, W = flow.size()
        tensor_hor = torch.linspace(-1.0, 1.0, W, device=feature.device, dtype=feature.dtype).view(
            1, 1, 1, W).expand(N, -1, H, -1)
        tensor_ver = torch.linspace(-1.0, 1.0, H, device=feature.device, dtype=feature.dtype).view(
            1, 1, H, 1).expand(N, -1, -1, W)
        backward_grid[device_id][str(flow.size())] = torch.cat([tensor_hor, tensor_ver], 1)

    flow = torch.cat([flow[:, 0:1, :, :] / ((feature.size(3) - 1.0) / 2.0),
                      flow[:, 1:2, :, :] / ((feature.size(2) - 1.0) / 2.0)], 1)

    grid = (backward_grid[device_id][str(flow.size())] + flow)
    return torch.nn.functional.grid_sample(input=feature,
                                           grid=grid.permute(0, 2, 3, 1),
                                           mode=mode,
                                           padding_mode='border',
                                           align_corners=True)


def multi_flow_warp(feature, flows, weights=None, mode='bilinear'):
    """
    多光流图补偿 + 加权融合
    Args:
        feature: 输入特征图 [N, C, H, W]
        flows: k个光流图的列表（每个为 [N, 2, H, W]）或张量 [N, k, 2, H, W]
        weights: 权重张量 [N, k, 1, 1] 或 None（均值融合）
        mode: 插值模式 ('bilinear' 或 'nearest')
    Returns:
        加权融合后的特征图 [N, C, H, W]
    """
    # 统一输入格式：将flows转为 [N, k, 2, H, W]

    N, _, H, W = flows.shape
    flows = flows.reshape(N, -1, 2, H, W)
    N, k, _, H, W = flows.shape
    # 生成所有光流图的采样网格 [N, k, H, W, 2]
    device_id = -1 if feature.device == torch.device('cpu') else feature.device.index
    if str((N, H, W)) not in backward_grid[device_id]:
        tensor_hor = torch.linspace(-1.0, 1.0, W, device=feature.device, dtype=feature.dtype).view(
            1, 1, 1, W).expand(N, -1, H, -1)
        tensor_ver = torch.linspace(-1.0, 1.0, H, device=feature.device, dtype=feature.dtype).view(
            1, 1, H, 1).expand(N, -1, -1, W)
        backward_grid[device_id][str((N, H, W))] = torch.cat([tensor_hor, tensor_ver], 1)

    # 归一化光流（与原始torch_warp一致）
    # flow = torch.cat([flows[:, 0:1, :, :] / ((feature.size(3) - 1.0) / 2.0),
    #                   flows[:, 1:2, :, :] / ((feature.size(2) - 1.0) / 2.0)], 1)
    flows = torch.cat(
        [flows[:, :, 0:1] / ((feature.size(3) - 1.0) / 2.0), flows[:, :, 1:2] / ((feature.size(2) - 1.0) / 2.0)],
        dim=2)  # [N, k, 2, H, W]

    # 生成所有网格 [N, k, H, W, 2]
    base_grid = backward_grid[device_id][str((N, H, W))]  # [N, 2, H, W]
    grids = base_grid.unsqueeze(1).permute(0, 1, 3, 4, 2) + flows.permute(0, 1, 3, 4, 2)  # [N, k, H, W, 2]

    # 批量grid_sample（通过reshape合并k维度）
    warped = torch.nn.functional.grid_sample(
        input=feature.unsqueeze(1).expand(-1, k, -1, -1, -1).reshape(N * k, -1, H, W),
        grid=grids.reshape(N * k, H, W, 2),
        mode=mode,
        padding_mode='border',
        align_corners=True
    )  # [N*k, C, H, W]
    warped = warped.view(N, k, -1, H, W)  # [N, k, C, H, W]

    # 加权融合
    if weights is None:
        output = warped.mean(dim=1)  # 均值融合 [N, C, H, W]
    else:
        weights = weights.view(N, k, 1, 1, 1)  # [N, k, 1, 1, 1]
        output = (warped * weights).sum(dim=1)  # [N, C, H, W]

    return output


def flow_warp(im, flow, mode='bilinear'):
    warp = torch_warp(im, flow, mode)
    return warp


def bilinearupsacling(inputfeature):
    inputheight = inputfeature.size()[2]
    inputwidth = inputfeature.size()[3]
    outfeature = F.interpolate(
        inputfeature, (inputheight * 2, inputwidth * 2), mode='bilinear', align_corners=False)
    return outfeature


def bilineardownsacling(inputfeature):
    inputheight = inputfeature.size()[2]
    inputwidth = inputfeature.size()[3]
    outfeature = F.interpolate(
        inputfeature, (inputheight // 2, inputwidth // 2), mode='bilinear', align_corners=False)
    return outfeature


class ResBlock(nn.Module):
    def __init__(self, channel, slope=0.01, start_from_relu=True, end_with_relu=False,
                 bottleneck=False):
        super().__init__()
        self.relu = nn.LeakyReLU(negative_slope=slope)
        if slope < 0.0001:
            self.relu = nn.ReLU()
        if bottleneck:
            self.conv1 = nn.Conv2d(channel, channel // 2, 3, padding=1)
            self.conv2 = nn.Conv2d(channel // 2, channel, 3, padding=1)
        else:
            self.conv1 = nn.Conv2d(channel, channel, 3, padding=1)
            self.conv2 = nn.Conv2d(channel, channel, 3, padding=1)
        self.first_layer = self.relu if start_from_relu else nn.Identity()
        self.last_layer = self.relu if end_with_relu else nn.Identity()

    def forward(self, x):
        out = self.first_layer(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.last_layer(out)
        return x + out


modelspath = './flow_pretrain_np/'


def loadweightformnp(layername):
    index = layername.find('modelL')
    if index == -1:
        print('laod models error!!')
    else:
        name = layername[index:index + 11]
        modelweight = modelspath + name + '-weight.npy'
        modelbias = modelspath + name + '-bias.npy'
        weightnp = np.load(modelweight)
        # weightnp = np.transpose(weightnp, [2, 3, 1, 0])
        # print(weightnp)
        biasnp = np.load(modelbias)
        return torch.from_numpy(weightnp), torch.from_numpy(biasnp)


class MEBasic(nn.Module):
    '''
    Get flow
    '''

    def __init__(self, layername):
        super(MEBasic, self).__init__()

        self.conv1 = nn.Conv2d(8, 32, 7, 1, padding=3)

        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 7, 1, padding=3)

        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 32, 7, 1, padding=3)

        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(32, 16, 7, 1, padding=3)

        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(16, 2, 7, 1, padding=3)



    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.conv5(x)
        return x


class ME_Spynet(nn.Module):
    '''
    Get flow
    '''

    def __init__(self, layername='motion_estimation'):
        super(ME_Spynet, self).__init__()
        self.L = 4
        self.moduleBasic = torch.nn.ModuleList(
            [MEBasic(layername + 'modelL' + str(intLevel + 1)) for intLevel in range(4)])

    def forward(self, im1, im2):
        batchsize = im1.size()[0]
        im1_pre = im1
        im2_pre = im2

        im1list = [im1_pre]
        im2list = [im2_pre]
        for intLevel in range(self.L - 1):
            im1list.append(F.avg_pool2d(
                im1list[intLevel], kernel_size=2, stride=2))
            im2list.append(F.avg_pool2d(
                im2list[intLevel], kernel_size=2, stride=2))

        shape_fine = im2list[self.L - 1].size()
        zeroshape = [batchsize, 2, shape_fine[2] // 2, shape_fine[3] // 2]
        device = im1.device
        flowfileds = torch.zeros(
            zeroshape, dtype=torch.float32, device=device)
        for intLevel in range(self.L):
            flowfiledsUpsample = bilinearupsacling(flowfileds) * 2.0
            flowfileds = flowfiledsUpsample + \
                         self.moduleBasic[intLevel](torch.cat([im1list[self.L - 1 - intLevel],
                                                               flow_warp(im2list[self.L - 1 - intLevel],
                                                                         flowfiledsUpsample),
                                                               flowfiledsUpsample], 1))

        return flowfileds


class OffsetBasic(nn.Module):
    '''
    Get flow
    '''

    def __init__(self, in_c, scale=2, layername=None):
        super(OffsetBasic, self).__init__()
        self.conv1 = nn.Conv2d(in_c, 32, 7, 1, padding=3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 7, 1, padding=3)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 32, 7, 1, padding=3)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(32, 16, 7, 1, padding=3)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(16, 18, 3, 1, padding=1)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        # if hasattr(self,'attention'):
        #     x = self.attention(x)
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.conv5(x)
        return x


class DualFlowPyramid(nn.Module):
    def __init__(self, num_scales=4):
        super(DualFlowPyramid, self).__init__()
        self.num_scales = num_scales
        self.moduleBasic = torch.nn.ModuleList(
            [MEBasic('motion_estimation' + 'modelL' + str(intLevel + 1)) for intLevel in range(4)])

        in_channels = 32
        self.feature_extract = nn.Sequential(nn.Conv2d(3, in_channels, 5, 1, 2), nn.LeakyReLU())

        self.off_convs = nn.ModuleList([
            OffsetBasic(18 + 11+in_channels, i, 'motion_estimation' + 'modelL' + str(i + 1))
            if i < 2 else OffsetBasic(18 + 11, i, 'motion_estimation' + 'modelL' + str(i + 1))
            for i in range(num_scales)
        ])

        self.down_convs_cur = nn.ModuleList([
            nn.Sequential(conv3x3(in_channels, in_channels, 2), nn.LeakyReLU(0.2)) for _ in range(num_scales - 1)
        ])
        self.down_convs_ref = nn.ModuleList([
            nn.Sequential(conv3x3(in_channels, in_channels, 2), nn.LeakyReLU(0.2)) for _ in range(num_scales - 1)
        ])
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        self.sigmoid = torch.nn.Sigmoid()
        self.dcn_last = df.DeformConv2d(3, 3, kernel_size=3, padding=1)

        self.cpa = nn.ModuleList(
            [GobalSuperPixelAttention(dim=in_channels, superpixel_size=[8, 8]) for _ in range(num_scales - 2)
             ])

    def forward(self, x_cur, x_ref):
        feature_ref = self.feature_extract(x_ref)
        feature_cur = self.feature_extract(x_cur)
        feat1list = [feature_cur]
        feat2list = [feature_ref]
        im1_pre = x_cur
        im2_pre = x_ref

        im1list = [im1_pre]
        im2list = [im2_pre]
        for idx in range(self.num_scales - 1):
            feat1list.append(self.down_convs_cur[idx](feat1list[idx]))
            feat2list.append(self.down_convs_ref[idx](feat2list[idx]))
            im1list.append(F.avg_pool2d(im1list[idx], kernel_size=2, stride=2))
            im2list.append(F.avg_pool2d(im2list[idx], kernel_size=2, stride=2))

        shape_fine = im2list[self.num_scales - 1].size()
        zeroshape = [shape_fine[0], 18, shape_fine[2] // 2, shape_fine[3] // 2]
        offset = torch.zeros(zeroshape, dtype=torch.float32).to(x_cur)
        zeroshape = [shape_fine[0], 2, shape_fine[2] // 2, shape_fine[3] // 2]
        flowfileds = torch.zeros(zeroshape, dtype=torch.float32).to(x_cur)

        for intLevel in range(self.num_scales):
            flowfiledsUpsample = bilinearupsacling(flowfileds) * 2.0
            warp_frame = multi_flow_warp(im2list[self.num_scales - 1 - intLevel], flowfiledsUpsample)
            flowfileds = flowfiledsUpsample + self.moduleBasic[intLevel](
                torch.cat([im1list[self.num_scales - 1 - intLevel], warp_frame, flowfiledsUpsample], 1))

            offset_upsample = self.upsample(offset) * 2
            warp_feature = multi_flow_warp(im2list[self.num_scales - 1 - intLevel], offset_upsample)
            if intLevel < self.num_scales - 2:
                sp_corr = self.cpa[intLevel](feat1list[self.num_scales - 1 - intLevel], feat2list[self.num_scales - 1 - intLevel])
                offset_feature = torch.cat((im1list[self.num_scales - 1 - intLevel], warp_feature, warp_frame,
                                            sp_corr, offset_upsample, flowfiledsUpsample), 1)
            else:
                offset_feature = torch.cat((im1list[self.num_scales - 1 - intLevel], warp_feature, warp_frame,
                                            offset_upsample, flowfiledsUpsample), 1)
            offset = offset_upsample + self.off_convs[intLevel](offset_feature)
        return torch.cat([flowfileds, offset], dim=1)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = torch.mean(x, dim=(-1, -2))
        y = self.fc(y)
        return x * y[:, :, None, None]


class ConvBlockResidual(nn.Module):
    def __init__(self, ch_in, ch_out, se_layer=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1),
            SELayer(ch_out) if se_layer else nn.Identity(),
        )
        self.up_dim = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.up_dim(x)
        return x2 + x1


class UNet(nn.Module):
    def __init__(self, in_ch=64, out_ch=64):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = ConvBlockResidual(ch_in=in_ch, ch_out=32)
        self.conv2 = ConvBlockResidual(ch_in=32, ch_out=64)
        self.conv3 = ConvBlockResidual(ch_in=64, ch_out=128)

        self.context_refine = nn.Sequential(
            ResBlock(128, 0),
            ResBlock(128, 0),
            ResBlock(128, 0),
            ResBlock(128, 0),
        )

        self.up3 = subpel_conv1x1(128, 64, 2)
        self.up_conv3 = ConvBlockResidual(ch_in=128, ch_out=64)

        self.up2 = subpel_conv1x1(64, 32, 2)
        self.up_conv2 = ConvBlockResidual(ch_in=64, ch_out=out_ch)

    def forward(self, x):
        # encoding path
        x1 = self.conv1(x)
        x2 = self.max_pool(x1)

        x2 = self.conv2(x2)
        x3 = self.max_pool(x2)

        x3 = self.conv3(x3)
        x3 = self.context_refine(x3)

        # decoding + concat path
        d3 = self.up3(x3)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.up_conv2(d2)
        return d2


def get_enc_dec_models(input_channel, output_channel, channel):
    enc = nn.Sequential(
        ResidualBlockWithStride(input_channel, channel, stride=2),
        ResidualBlock(channel, channel),
        ResidualBlockWithStride(channel, channel, stride=2),
        ResidualBlock(channel, channel),
        ResidualBlockWithStride(channel, channel, stride=2),
        ResidualBlock(channel, channel),
        conv3x3(channel, channel, stride=2),
    )

    dec = nn.Sequential(
        ResidualBlock(channel, channel),
        ResidualBlockUpsample(channel, channel, 2),
        ResidualBlock(channel, channel),
        ResidualBlockUpsample(channel, channel, 2),
        ResidualBlock(channel, channel),
        ResidualBlockUpsample(channel, channel, 2),
        ResidualBlock(channel, channel),
        subpel_conv1x1(channel, output_channel, 2),
    )

    return enc, dec


def get_hyper_enc_dec_models(y_channel, z_channel):
    enc = nn.Sequential(
        conv3x3(y_channel, z_channel),
        nn.LeakyReLU(),
        conv3x3(z_channel, z_channel),
        nn.LeakyReLU(),
        conv3x3(z_channel, z_channel, stride=2),
        nn.LeakyReLU(),
        conv3x3(z_channel, z_channel),
        nn.LeakyReLU(),
        conv3x3(z_channel, z_channel, stride=2),
    )

    dec = nn.Sequential(
        conv3x3(z_channel, y_channel),
        nn.LeakyReLU(),
        subpel_conv1x1(y_channel, y_channel, 2),
        nn.LeakyReLU(),
        conv3x3(y_channel, y_channel * 3 // 2),
        nn.LeakyReLU(),
        subpel_conv1x1(y_channel * 3 // 2, y_channel * 3 // 2, 2),
        nn.LeakyReLU(),
        conv3x3(y_channel * 3 // 2, y_channel * 2),
    )

    return enc, dec
