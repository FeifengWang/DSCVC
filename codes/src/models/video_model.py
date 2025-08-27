# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import time
from typing import List

import torch

from torch import nn

from .common_model import CompressionModel
from .i2c_utils import IntraCtxBlock
from .video_net import ME_Spynet, flow_warp, ResBlock, bilineardownsacling, LowerBound, UNet, \
    get_enc_dec_models, get_hyper_enc_dec_models, DualFlowPyramid, multi_flow_warp
from ..layers.layers import conv3x3, subpel_conv1x1, subpel_conv3x3
from ..utils.stream_helper import get_downsampled_shape, encode_p, decode_p, filesize, \
    get_rounded_q, get_state_dict


class FeatureExtractor(nn.Module):
    def __init__(self, channel=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channel, channel, 3, stride=1, padding=1)
        self.res_block1 = ResBlock(channel)
        self.conv2 = nn.Conv2d(channel, channel, 3, stride=2, padding=1)
        self.res_block2 = ResBlock(channel)
        self.conv3 = nn.Conv2d(channel, channel, 3, stride=2, padding=1)
        self.res_block3 = ResBlock(channel)

    def forward(self, feature):
        layer1 = self.conv1(feature)
        layer1 = self.res_block1(layer1)

        layer2 = self.conv2(layer1)
        layer2 = self.res_block2(layer2)

        layer3 = self.conv3(layer2)
        layer3 = self.res_block3(layer3)

        return layer1, layer2, layer3


class MultiScaleContextFusion(nn.Module):
    def __init__(self, channel_in=64, channel_out=64):
        super().__init__()
        self.conv3_up = subpel_conv3x3(channel_in, channel_out, 2)
        self.res_block3_up = ResBlock(channel_out)
        self.conv3_out = nn.Conv2d(channel_out, channel_out, 3, padding=1)
        self.res_block3_out = ResBlock(channel_out)
        self.conv2_up = subpel_conv3x3(channel_out * 2, channel_out, 2)
        self.res_block2_up = ResBlock(channel_out)
        self.conv2_out = nn.Conv2d(channel_out * 2, channel_out, 3, padding=1)
        self.res_block2_out = ResBlock(channel_out)
        self.conv1_out = nn.Conv2d(channel_out * 2, channel_out, 3, padding=1)
        self.res_block1_out = ResBlock(channel_out)

    def forward(self, context1, context2, context3):
        context3_up = self.conv3_up(context3)
        context3_up = self.res_block3_up(context3_up)
        context3_out = self.conv3_out(context3)
        context3_out = self.res_block3_out(context3_out)
        context2_up = self.conv2_up(torch.cat((context3_up, context2), dim=1))
        context2_up = self.res_block2_up(context2_up)
        context2_out = self.conv2_out(torch.cat((context3_up, context2), dim=1))
        context2_out = self.res_block2_out(context2_out)
        context1_out = self.conv1_out(torch.cat((context2_up, context1), dim=1))
        context1_out = self.res_block1_out(context1_out)
        context1 = context1 + context1_out
        context2 = context2 + context2_out
        context3 = context3 + context3_out
        return context1, context2, context3


class InterCtxBlock(nn.Module):
    def  __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),nn.LeakyReLU(0.2))
        self.conv2 = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x, context):
        sim = torch.cat([x, context], dim=1)
        sim = self.conv1(sim)
        sim = self.conv2(sim)
        return torch.sigmoid(sim)

class ContextualEncoder(nn.Module):
    def __init__(self, channel_N=64, channel_M=96):
        super().__init__()
        self.conv0 = nn.Conv2d(3, channel_N, 3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(channel_N * 2, channel_N, 3, stride=2, padding=1)
        self.res_1 = ResBlock(channel_N, bottleneck=True, slope=0.1,start_from_relu=True, end_with_relu=True)
        self.conv2 = nn.Conv2d(channel_N * 2, channel_N, 3, stride=2, padding=1)
        self.res_2 = ResBlock(channel_N, bottleneck=True, slope=0.1,start_from_relu=True, end_with_relu=True)
        self.res_3 = ResBlock(channel_N, bottleneck=True, slope=0.1,start_from_relu=True, end_with_relu=True)
        self.conv3 = nn.Conv2d(channel_N * 2, channel_N, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(channel_N, channel_M, 3, stride=2, padding=1)
        self.calc_sim1 = InterCtxBlock(channel_N)
        self.calc_sim2 = InterCtxBlock(channel_N)
        self.calc_sim3 = InterCtxBlock(channel_N)
        self.det2 = nn.Sequential(
            *[IntraCtxBlock(channel_N, channel_N, 16, 8, 0, 'W' if not i % 2 else 'SW')
              for i in range(1)])
        self.det3 = nn.Sequential(
            *[IntraCtxBlock(channel_N, channel_N, 16, 8, 0, 'W' if not i % 2 else 'SW')
              for i in range(1)])
    def forward(self, x, context1, context2, context3):
        feature = self.conv0(x)
        g1 = self.calc_sim1(feature, context1)
        context1 = g1*context1
        feature = self.conv_1(torch.cat([feature, context1], dim=1))
        feature = self.res_1(feature)
        g2 = self.calc_sim2(feature, context2) * context2
        context2 = g2*context2
        feature = self.det2[0](torch.cat([feature, context2], dim=1),g2)
        feature = self.conv2(feature)
        feature = self.res_2(feature)
        g3 = self.calc_sim3(feature, context3)
        context3 = g3 * context3
        feature = self.det3[0](torch.cat([feature, context3], dim=1),g3)
        feature = self.conv3(feature)
        feature = self.res_3(feature)
        feature = self.conv4(feature)
        return feature
    def calculate_similarity(self, x, context):
        # 利用余弦相似性作为相似性测量
        x_flat = x.view(x.size(0), -1)
        context_flat = context.view(context.size(0), -1)
        sim = torch.nn.functional.cosine_similarity(x_flat, context_flat, dim=1)
        return sim.view(-1, 1, 1, 1)


class ContextualDecoder(nn.Module):
    def __init__(self, channel_N=64, channel_M=96):
        super().__init__()
        self.up1 = subpel_conv3x3(channel_M, channel_N, 2)
        self.up2 = subpel_conv3x3(channel_N, channel_N, 2)
        self.res_1 = ResBlock(channel_N, bottleneck=True, slope=0.1, start_from_relu=True, end_with_relu=True)
        self.up3 = subpel_conv3x3(channel_N * 2, channel_N, 2)
        self.res_2 = ResBlock(channel_N, bottleneck=True, slope=0.1,start_from_relu=True, end_with_relu=True)
        self.res_3 = ResBlock(channel_N, bottleneck=True, slope=0.1,start_from_relu=True, end_with_relu=True)
        self.up_4 = subpel_conv3x3(channel_N * 2, 64, 2)
        self.calc_sim2 = InterCtxBlock(channel_N)
        self.calc_sim3 = InterCtxBlock(channel_N)
        self.ddt2 = nn.Sequential(
            *[IntraCtxBlock(channel_N, channel_N, 16, 8, 0, 'W' if not i % 2 else 'SW')
              for i in range(1)])
        self.ddt3 = nn.Sequential(
            *[IntraCtxBlock(channel_N, channel_N, 16, 8, 0, 'W' if not i % 2 else 'SW')
              for i in range(1)])
    def forward(self, x, context2, context3):
        feature = self.up1(x)
        feature = self.res_1(feature)
        feature = self.up2(feature)
        feature = self.res_2(feature)
        g = self.calc_sim2(feature, context3)
        context3 = g * context3
        feature = self.ddt3[0](torch.cat([feature, context3], dim=1), g)

        feature = self.up3(feature)
        feature = self.res_3(feature)
        g2 = self.calc_sim3(feature, context2)
        context2 = g2 * context2
        feature = self.ddt2[0](torch.cat([feature, context2], dim=1),g2)

        feature = self.up_4(feature)
        return feature



class ReconGeneration(nn.Module):
    def __init__(self, ctx_channel=64, res_channel=64, channel=64):
        super().__init__()
        self.first_conv0 = nn.Conv2d(ctx_channel + res_channel, channel, 3, stride=1, padding=1)
        self.unet_1 = UNet(channel)
        self.unet_2 = UNet(channel)
        self.recon_conv = nn.Conv2d(channel, 3, 3, stride=1, padding=1)
        self.calc_sim1 = InterCtxBlock(channel)
    def forward(self, res, ctx):
        ctx = self.calc_sim1(res,ctx) * ctx
        feature = self.first_conv0(torch.cat((res,ctx), dim=1))
        feature = self.unet_1(feature)
        feature = self.unet_2(feature)
        recon = self.recon_conv(feature)
        return feature, recon

class DSCVC(CompressionModel):
    def __init__(self, anchor_num=4):
        super().__init__(y_distribution='laplace', z_channel=64, mv_z_channel=64)
        channel_mv = 64
        channel_N = 64
        channel_M = 96

        self.channel_mv = channel_mv
        self.channel_N = channel_N
        self.channel_M = channel_M
        self.optic_flow = DualFlowPyramid()
        self.mv_encoder, self.mv_decoder = get_enc_dec_models(20, 20, channel_mv)
        self.mv_hyper_prior_encoder, self.mv_hyper_prior_decoder = \
            get_hyper_enc_dec_models(channel_mv, channel_N)

        self.mv_y_prior_fusion = nn.Sequential(
            nn.Conv2d(channel_mv * 3, channel_mv * 3, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel_mv * 3, channel_mv * 3, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel_mv * 3, channel_mv * 3, 3, stride=1, padding=1)
        )

        self.mv_y_spatial_prior = nn.Sequential(
            nn.Conv2d(channel_mv * 4, channel_mv * 3, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel_mv * 3, channel_mv * 3, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel_mv * 3, channel_mv * 2, 3, padding=1)
        )

        self.feature_adaptor_I = nn.Conv2d(3, channel_N, 3, stride=1, padding=1)
        self.feature_adaptor_P = nn.Conv2d(channel_N, channel_N, 1)
        self.feature_extractor = FeatureExtractor()
        self.context_fusion_net = MultiScaleContextFusion()

        self.contextual_encoder = ContextualEncoder(channel_N=channel_N, channel_M=channel_M)

        self.contextual_hyper_prior_encoder = nn.Sequential(
            nn.Conv2d(channel_M, channel_N, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel_N, channel_N, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel_N, channel_N, 3, stride=2, padding=1),
        )

        self.contextual_hyper_prior_decoder = nn.Sequential(
            conv3x3(channel_N, channel_M),
            nn.LeakyReLU(),
            subpel_conv1x1(channel_M, channel_M, 2),
            nn.LeakyReLU(),
            conv3x3(channel_M, channel_M * 3 // 2),
            nn.LeakyReLU(),
            subpel_conv1x1(channel_M * 3 // 2, channel_M * 3 // 2, 2),
            nn.LeakyReLU(),
            conv3x3(channel_M * 3 // 2, channel_M * 2),
        )

        self.temporal_prior_encoder = nn.Sequential(
            nn.Conv2d(channel_N, channel_M * 3 // 2, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(channel_M * 3 // 2, channel_M * 2, 3, stride=2, padding=1),
        )

        self.y_prior_fusion = nn.Sequential(
            nn.Conv2d(channel_M * 5, channel_M * 4, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel_M * 4, channel_M * 3, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel_M * 3, channel_M * 3, 3, stride=1, padding=1)
        )

        self.y_spatial_prior = nn.Sequential(
            nn.Conv2d(channel_M * 4, channel_M * 3, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel_M * 3, channel_M * 3, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel_M * 3, channel_M * 2, 3, padding=1)
        )

        self.contextual_decoder = ContextualDecoder(channel_N=channel_N, channel_M=channel_M)
        self.recon_generation_net = ReconGeneration()

        self.mv_y_q_basic = nn.Parameter(torch.ones((1, channel_mv, 1, 1)))
        self.mv_y_q_scale = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))
        self.y_q_basic = nn.Parameter(torch.ones((1, channel_M, 1, 1)))
        self.y_q_scale = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))
        self.anchor_num = int(anchor_num)

    def multi_scale_feature_extractor(self, dpb):
        if dpb["ref_feature"] is None:
            feature = self.feature_adaptor_I(dpb["ref_frame"])
        else:
            feature = self.feature_adaptor_P(dpb["ref_feature"])
        return self.feature_extractor(feature)

    def motion_compensation(self, dpb, mv):
        warpframe = multi_flow_warp(dpb["ref_frame"], mv)
        mv2 = bilineardownsacling(mv) / 2
        mv3 = bilineardownsacling(mv2) / 2
        ref_feature1, ref_feature2, ref_feature3 = self.multi_scale_feature_extractor(dpb)
        context1 = multi_flow_warp(ref_feature1, mv)
        context2 = multi_flow_warp(ref_feature2, mv2)
        context3 = multi_flow_warp(ref_feature3, mv3)
        context1, context2, context3 = self.context_fusion_net(context1, context2, context3)
        return context1, context2, context3, warpframe

    @staticmethod
    def get_q_scales_from_ckpt(ckpt_path):
        ckpt = get_state_dict(ckpt_path)
        y_q_scales = ckpt["y_q_scale"]
        mv_y_q_scales = ckpt["mv_y_q_scale"]
        return y_q_scales.reshape(-1), mv_y_q_scales.reshape(-1)

    def get_curr_mv_y_q(self, q_scale):
        q_basic = LowerBound.apply(self.mv_y_q_basic, 0.5)
        return q_basic * q_scale

    def get_curr_y_q(self, q_scale):
        q_basic = LowerBound.apply(self.y_q_basic, 0.5)
        return q_basic * q_scale


    def forward_inter(self, x, dpb, mv_y_q_scale=None, y_q_scale=None):
        ref_frame = dpb["ref_frame"]
        curr_mv_y_q = self.get_curr_mv_y_q(mv_y_q_scale)
        curr_y_q = self.get_curr_y_q(y_q_scale)

        est_mv = self.optic_flow(x, ref_frame)
        mv_y = self.mv_encoder(est_mv)
        mv_y = mv_y / curr_mv_y_q
        mv_z = self.mv_hyper_prior_encoder(mv_y)
        mv_z_hat = self.quant(mv_z)
        mv_params = self.mv_hyper_prior_decoder(mv_z_hat)
        ref_mv_y = dpb["ref_mv_y"]
        if ref_mv_y is None:
            ref_mv_y = torch.zeros_like(mv_y)
        mv_params = torch.cat((mv_params, ref_mv_y), dim=1)
        mv_q_step, mv_scales, mv_means = self.mv_y_prior_fusion(mv_params).chunk(3, 1)
        mv_y_res, mv_y_q, mv_y_hat, mv_scales_hat = self.forward_dual_prior(
            mv_y, mv_means, mv_scales, mv_q_step, self.mv_y_spatial_prior)
        mv_y_hat = mv_y_hat * curr_mv_y_q

        mv_hat = self.mv_decoder(mv_y_hat)
        context1, context2, context3, warp_frame = self.motion_compensation(dpb, mv_hat)

        y = self.contextual_encoder(x, context1, context2, context3)
        y = y / curr_y_q
        z = self.contextual_hyper_prior_encoder(y)
        z_hat = self.quant(z)
        hierarchical_params = self.contextual_hyper_prior_decoder(z_hat)
        temporal_params = self.temporal_prior_encoder(context3)

        ref_y = dpb["ref_y"]
        if ref_y is None:
            ref_y = torch.zeros_like(y)
        params = torch.cat((temporal_params, hierarchical_params, ref_y), dim=1)
        q_step, scales, means = self.y_prior_fusion(params).chunk(3, 1)
        y_res, y_q, y_hat, scales_hat = self.forward_dual_prior(
            y, means, scales, q_step, self.y_spatial_prior)
        y_hat = y_hat * curr_y_q

        recon_image_feature = self.contextual_decoder(y_hat, context2, context3)
        feature, recon_image = self.recon_generation_net(recon_image_feature, context1)

        B, _, H, W = x.size()
        pixel_num = B * H * W
        mse = self.mse(x, recon_image)
        ssim = self.ssim(x, recon_image)
        me_mse = self.mse(x, warp_frame)
        mse = torch.sum(mse) / pixel_num
        me_mse = torch.sum(me_mse) / pixel_num

        if self.training:
            y_for_bit = self.add_noise(y_res)
            mv_y_for_bit = self.add_noise(mv_y_res)
            z_for_bit = self.add_noise(z)
            mv_z_for_bit = self.add_noise(mv_z)
        else:
            y_for_bit = y_q
            mv_y_for_bit = mv_y_q
            z_for_bit = z_hat
            mv_z_for_bit = mv_z_hat
        bits_y = self.get_y_laplace_bits(y_for_bit, scales_hat)
        bits_mv_y = self.get_y_laplace_bits(mv_y_for_bit, mv_scales_hat)
        bits_z = self.get_z_bits(z_for_bit, self.bit_estimator_z)
        bits_mv_z = self.get_z_bits(mv_z_for_bit, self.bit_estimator_z_mv)

        bpp_y = torch.sum(bits_y) / pixel_num
        bpp_z = torch.sum(bits_z) / pixel_num
        bpp_mv_y = torch.sum(bits_mv_y) / pixel_num
        bpp_mv_z = torch.sum(bits_mv_z) / pixel_num


        bpp = bpp_y + bpp_z + bpp_mv_y + bpp_mv_z
        bit = torch.sum(bpp) * pixel_num
        bit_y = torch.sum(bpp_y) * pixel_num
        bit_z = torch.sum(bpp_z) * pixel_num
        bit_mv_y = torch.sum(bpp_mv_y) * pixel_num
        bit_mv_z = torch.sum(bpp_mv_z) * pixel_num
        return {"bpp_mv_y": bpp_mv_y,
                "bpp_mv_z": bpp_mv_z,
                "bpp_y": bpp_y,
                "bpp_z": bpp_z,
                "bpp": bpp,
                "me_mse": me_mse,
                "mse": mse,
                "ssim": ssim,
                "recon_image": recon_image,
                "warp_frame": warp_frame,
                "dpb": {
                    "ref_frame": recon_image,
                    "ref_feature": feature,
                    "ref_y": y_hat,
                    "ref_mv_y": mv_y_hat,
                },
                "bit": bit,
                "bit_y": bit_y,
                "bit_z": bit_z,
                "bit_mv_y": bit_mv_y,
                "bit_mv_z": bit_mv_z,
                }

    def forward(self, frames, rate_idx=0):
        if not isinstance(frames, List):
            raise RuntimeError(f"Invalid number of frames: {len(frames)}.")

        reconstructions = []
        warp_predictions = []
        weighted_list = [0.5, 1.2, 0.5, 0.9]
        sum_mse_loss = 0
        sum_warploss = 0
        weighted_mse_loss = 0
        sum_bpp_feature = 0
        sum_bpp_z, sum_bpp_mv, sum_bpp_mv_z, sum_bpp = 0, 0, 0, 0

        dpb = {
            "ref_frame": frames[0],
            "ref_feature": None,
            "ref_y": None,
            "ref_mv_y": None,
        }

        for i in range(1, len(frames)):
            x = frames[i]# B,3,W,H

            result_dict = self.forward_inter(x, dpb, self.mv_y_q_scale[rate_idx], self.y_q_scale[rate_idx])
            dpb = result_dict["dpb"]
            x_ref = result_dict["recon_image"]
            mse_loss = result_dict["mse"]
            warploss = result_dict["me_mse"]
            bpp_feature = result_dict["bpp_y"]
            bpp_z = result_dict["bpp_z"]
            bpp_mv = result_dict["bpp_mv_y"]
            bpp_mv_z = result_dict["bpp_mv_z"]
            bpp = result_dict["bpp"]
            reconstructions.append(x_ref)
            warp_predictions.append(result_dict['warp_frame'])
            sum_mse_loss += mse_loss
            sum_warploss += warploss
            weighted_mse_loss += weighted_list[i % 4] * mse_loss
            sum_bpp_feature += bpp_feature
            sum_bpp_z += bpp_z
            sum_bpp_mv += bpp_mv
            sum_bpp_mv_z += bpp_mv_z
            sum_bpp += bpp
        avg_mse_loss = sum_mse_loss.mean()/(len(frames)-1)
        avg_warploss = sum_warploss.mean()/(len(frames)-1)
        weighted_mse_loss = weighted_mse_loss.mean()/(len(frames)-1)
        avg_bpp_feature = sum_bpp_feature.mean()/(len(frames)-1)
        avg_bpp_z = sum_bpp_z.mean()/(len(frames)-1)
        avg_bpp_mv_z = sum_bpp_mv_z.mean()/(len(frames)-1)
        avg_bpp_mv,avg_bpp = sum_bpp_mv.mean()/(len(frames)-1), sum_bpp.mean()/(len(frames)-1)
        return {
            "x_hat": reconstructions,
            "x_warp": warp_predictions,
            "mse_loss": avg_mse_loss,
            "weighted_mse_loss": weighted_mse_loss,
            "warploss": avg_warploss,
            "bpp_y": avg_bpp_feature,
            "bpp_z": avg_bpp_z,
            "bpp_mv_y": avg_bpp_mv,
            "bpp_mv_z": avg_bpp_mv_z,
            "bpp": avg_bpp,
        }

