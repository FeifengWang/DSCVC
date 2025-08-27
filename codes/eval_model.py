# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import json
import math
import os.path
import sys
import time

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import mean

from pytorch_msssim import ms_ssim
from torch import Tensor
from torch.cuda import amp
from torch.utils.model_zoo import tqdm

import compressai

from compressai.datasets import RawVideoSequence, VideoFormat
from torchvision import transforms

from utils.yuv2rgb_utlis import (
    rgb2ycbcr,
    ycbcr2rgb,
    yuv_420_to_444,
    yuv_444_to_420,
)


from src.models.intra_model_scc import IntraModel
from src.models.video_model import DSCVC

Frame = Union[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, ...]]

RAWVIDEO_EXTENSIONS = (".yuv",)  # read raw yuv videos for now


def collect_videos(rootpath: str) -> List[str]:
    video_files = []
    for ext in RAWVIDEO_EXTENSIONS:
        video_files.extend(Path(rootpath).glob(f"*{ext}"))
    return video_files


def to_tensors(
    frame: Tuple[np.ndarray, np.ndarray, np.ndarray],
    max_value: int = 1,
    device: str = "cpu",
) -> Frame:
    return tuple(
        torch.from_numpy(np.true_divide(c, max_value, dtype=np.float32)).to(device)
        for c in frame
    )


def aggregate_results(filepaths: List[Path]) -> Dict[str, Any]:
    metrics = defaultdict(list)

    for f in filepaths:
        with f.open("r") as fd:
            data = json.load(fd)
        for k, v in data["results"].items():
            metrics[k].append(v)

    agg = {k: np.mean(v) for k, v in metrics.items()}
    return agg


def convert_yuv420_to_rgb(
    frame: Tuple[np.ndarray, np.ndarray, np.ndarray], device: torch.device, max_val: int
) -> Tensor:
    out = to_tensors(frame, device=str(device), max_value=max_val)
    out = yuv_420_to_444(
        tuple(c.unsqueeze(0).unsqueeze(0) for c in out), mode="bicubic"
    )
    return ycbcr2rgb(out)

def convert_yuv444_to_rgb(
    frame: Tuple[np.ndarray, np.ndarray, np.ndarray], device: torch.device, max_val: int
) -> Tensor:
    out = to_tensors(frame, device=str(device), max_value=max_val)
    out = torch.cat(tuple(c.unsqueeze(0).unsqueeze(0) for c in out), dim=1)
    return ycbcr2rgb(out)

def convert_rgb_to_yuv420(frame: Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return yuv_444_to_420(rgb2ycbcr(frame), mode="avg_pool")

def convert_rgb_to_yuv444(frame: Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    yuv = rgb2ycbcr(frame)
    if isinstance(yuv, torch.Tensor):
        y, u, v = yuv.chunk(3, 1)
    else:
        y, u, v = yuv
    return (y, u, v)

def pad(x: Tensor, p: int = 2 ** (4+2)) -> Tuple[Tensor, Tuple[int, ...]]:
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    padding = (padding_left, padding_right, padding_top, padding_bottom)
    x = F.pad(
        x,
        padding,
        mode="constant",
        value=0,
    )
    return x, padding


def crop(x: Tensor, padding: Tuple[int, ...]) -> Tensor:
    return F.pad(x, tuple(-p for p in padding))


def compute_metrics_for_frame(
    org_frame: Frame,
    rec_frame: Tensor,
    device: str = "cpu",
    max_val: int = 255,
    yuv_fmt: str =VideoFormat.YUV420,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    org_yuv = to_tensors(org_frame, device=str(device), max_value=max_val)
    org_yuv = tuple(p.unsqueeze(0).unsqueeze(0) for p in org_yuv)
    if yuv_fmt == VideoFormat.YUV420:
        rgb2yuv = convert_rgb_to_yuv420
        yuv2rgb = convert_yuv420_to_rgb
    else:
        rgb2yuv = convert_rgb_to_yuv444
        yuv2rgb = convert_yuv444_to_rgb
    rec_yuv = rgb2yuv(rec_frame)
    for i, component in enumerate("yuv"):
        org = (org_yuv[i] * max_val).clamp(0, max_val).round()
        rec = (rec_yuv[i] * max_val).clamp(0, max_val).round()
        out[f"psnr-{component}"] = 20 * np.log10(max_val) - 10 * torch.log10(
            (org - rec).pow(2).mean()
        )
    out["psnr-yuv"] = (4 * out["psnr-y"] + out["psnr-u"] + out["psnr-v"]) / 6

    org_rgb = yuv2rgb(org_frame, device, max_val)
    org_rgb = (org_rgb * max_val).clamp(0, max_val).round()
    rec_frame = (rec_frame * max_val).clamp(0, max_val).round()
    mse_rgb = (org_rgb - rec_frame).pow(2).mean()
    psnr_rgb = 20 * np.log10(max_val) - 10 * torch.log10(mse_rgb)

    ms_ssim_rgb = ms_ssim(org_rgb, rec_frame, data_range=max_val)
    out.update({"ms-ssim-rgb": ms_ssim_rgb, "mse-rgb": mse_rgb, "psnr-rgb": psnr_rgb})

    return out

def estimate_bits_frame(likelihoods) -> float:
    bpp = sum(
        (torch.log(lkl[k]).sum() / (-math.log(2)))
        for lkl in likelihoods.values()
        for k in ("y", "z")
    )
    return bpp



@torch.no_grad()
def eval_model_entropy_estimation(intra_net: nn.Module,
    net: nn.Module, sequence: Path, num_frames=0, gopsize=8, q_idx=3, y_q_scales=1, mv_y_q_scales=1, save_dir='', save_img=False
) -> Dict[str, Any]:
    org_seq = RawVideoSequence.from_file(str(sequence))

    device = next(net.parameters()).device
    if(num_frames==-1):
        num_frames = len(org_seq)

    max_val = 2**org_seq.bitdepth - 1
    results = defaultdict(list)

    print(f" encoding {sequence.stem}", file=sys.stderr)
    if org_seq.format == VideoFormat.YUV420:
        yuv2rgb = convert_yuv420_to_rgb
    elif org_seq.format == VideoFormat.YUV444:
        yuv2rgb = convert_yuv444_to_rgb
    else:
        print('Unsupported YUV Format!')
        return
    bpp_list=[]
    bpp_list_I=[]
    bpp_list_P=[]
    bpp_list_mv=[]
    bpp_list_ctx=[]
    refpsnr_list = []
    psnr_list_I = []
    psnr_list_P = []
    wpsnr_list = []
    recpsnr_list = []
    num_pixels = org_seq.height*org_seq.width

    debug_path = os.path.join(save_dir,'debug_info')
    os.makedirs(debug_path,exist_ok=True)
    debug_file = open(os.path.join(debug_path,sequence.name.replace('.yuv','.txt')),'w+')
    with tqdm(total=num_frames) as pbar:
        for i in range(num_frames):
            x_cur = yuv2rgb(org_seq[i], device, max_val)

            if (gopsize == -1 and i == 0) or (gopsize != -1 and i % gopsize == 0):
                x_cur, padding = pad(x_cur,p=2 ** (4+2+1))
                result = intra_net.forward_round(x_cur)
                # result = intra_net(x_cur)
                x_rec = crop(result["x_hat"], padding)
                bpp = sum(
                    (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                    for likelihoods in result["likelihoods"].values()
                )
                if 'plt_bits' in result:
                    bpp += result['plt_bits'] / num_pixels

                bpp = float(bpp.cpu().numpy())
                bpp_list.append(bpp)
                bpp_list_I.append(bpp)
                dpb = {
                    "ref_frame":pad(crop(result["x_hat"], padding),p=2 ** (4+2))[0],
                    "ref_feature": None,
                    "ref_y": None,
                    "ref_mv_y": None,
                }
                rec_psnr = 10 * torch.log10(1. / torch.nn.functional.mse_loss(x_cur, result["x_hat"]))
                INFO = f'frame_index {i} bpp:{bpp} rec_psnr:{rec_psnr.item()} \n'
                debug_file.write(INFO)
                psnr_list_I.append(rec_psnr.item())

            else:
                x_cur, padding = pad(x_cur,p=2 ** (4+2))
                x_ref = dpb["ref_frame"]
                result_dict = net.forward_inter(x_cur, dpb, mv_y_q_scale=mv_y_q_scales, y_q_scale=y_q_scales)
                dpb = result_dict["dpb"]
                x_rec = result_dict["dpb"]["ref_frame"]
                x_rec = crop(x_rec, padding)
                bpp_y = result_dict["bpp_y"].item()
                bpp_z = result_dict["bpp_z"].item()
                bpp_mv_y = result_dict["bpp_mv_y"].item()
                bpp_mv_z = result_dict["bpp_mv_z"].item()

                N, C, H, W = x_cur.size()
                pad_num_pixels = N * H * W
                bpp_motion = (bpp_mv_y + bpp_mv_z)*pad_num_pixels/num_pixels
                bpp_residual = (bpp_y + bpp_z)*pad_num_pixels/num_pixels
                bpp_list.append(bpp_motion+bpp_residual)
                bpp_list_P.append(bpp_motion+bpp_residual)
                warp_frame = result_dict["warp_frame"].clamp(0, 1)
                ref_psnr = 10 * torch.log10(1. / torch.nn.functional.mse_loss(x_cur, x_ref))
                warp_psnr = 10 * torch.log10(1. / torch.nn.functional.mse_loss(x_cur, warp_frame))
                rec_psnr = 10 * torch.log10(1. / torch.nn.functional.mse_loss(crop(x_cur, padding), x_rec))
                refpsnr_list.append(ref_psnr.item())
                wpsnr_list.append(warp_psnr.item())
                psnr_list_P.append(rec_psnr.item())
                bpp_list_mv.append(bpp_motion)
                bpp_list_ctx.append(bpp_residual)
                INFO=f'frame_index {i} bpp_motion:{bpp_motion:.5f} bpp_residual:{bpp_residual:.5f} bpp:{bpp_motion+bpp_residual:.5f} ref_psnr:{ref_psnr.item():.3f} warp_psnr:{warp_psnr.item():.3f} rec_psnr:{rec_psnr.item():.3f} \n'
                debug_file.write(INFO)

            x_rec = x_rec.clamp(0, 1)


            if save_img:
                path_name, file_name = os.path.split(sequence)
                seq_name = file_name.replace('.yuv', '')
                seq_path = os.path.join(save_dir, str(q_idx), seq_name)
                os.makedirs(seq_path, exist_ok=True)
                file_name = file_name.replace('.yuv', '_{:03d}.png'.format(i))
                trans1 = transforms.ToPILImage()
                img_rec = trans1(x_rec[0])
                img_rec.save(os.path.join(seq_path, file_name))
            metrics = compute_metrics_for_frame(
                org_seq[i],
                x_rec,
                device,
                max_val,
                org_seq.format,
            )

            for k, v in metrics.items():
                results[k].append(v)
            pbar.update(1)
    INFO = f'Average bpp:{np.mean(bpp_list):.5f} bpp_i:{np.mean(bpp_list_I):.5f} bpp_p:{np.mean(bpp_list_P):.5f} bpp_mv:{np.mean(bpp_list_mv):.5f} bpp_ctx:{np.mean(bpp_list_ctx):.5f} I_psnr:{np.mean(psnr_list_I):.3f} ref_psnr:{np.mean(refpsnr_list):.3f} warp_psnr:{np.mean(wpsnr_list):.3f} P_psnr:{np.mean(psnr_list_P):.3f} \n'
    debug_file.write(INFO)
    seq_results: Dict[str, Any] = {
        k: torch.mean(torch.stack(v)) for k, v in results.items()
    }

    seq_results["bpp"] = mean(bpp_list)
    seq_results["bitrate"] =seq_results["bpp"] * org_seq.width*org_seq.height*org_seq.framerate/1000

    for k, v in seq_results.items():
        if isinstance(v, torch.Tensor):
            seq_results[k] = v.item()
    return seq_results

def run_inference(
    filepaths,
    intra_net,
    net: nn.Module,
    q_idx,
    y_q_scales,
    mv_y_q_scales,
    outputdir: Path,
    force: bool = False,
    entropy_estimation: bool = False,
    trained_net: str = "",
    description: str = "",
    **args: Any,
) -> Dict[str, Any]:
    results_paths = []

    Path(outputdir).mkdir(parents=True, exist_ok=True)
    for filepath in sorted(filepaths):
        sequence_metrics_path = Path(outputdir) / f"{filepath.stem}-{trained_net}-q{q_idx}.json"
        results_paths.append(sequence_metrics_path)

        if force:
            sequence_metrics_path.unlink(missing_ok=True)
        if sequence_metrics_path.is_file():
            continue

        with amp.autocast(enabled=args["half"]):
            with torch.no_grad():
                if entropy_estimation:
                    metrics = eval_model_entropy_estimation(intra_net, net, filepath, args["frames"], args["gop"], q_idx, y_q_scales,mv_y_q_scales,outputdir, args["save"])
                else:
                    net.update(True)
                    pass
        with sequence_metrics_path.open("wb") as f:
            output = {
                "source": filepath.stem,
                "name": args["architecture"],
                "description": f"Inference ({description})",
                "results": metrics,
            }
            f.write(json.dumps(output, indent=2).encode())
    results = aggregate_results(results_paths)
    return results

def load_net(arch: str, checkpoint_path: str) -> nn.Module:
    state_dict = torch.load(checkpoint_path)
    state_dict = state_dict.get("state_dict", state_dict)
    net = DSCVC()
    net.load_state_dict(state_dict)
    net.eval()
    return net


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Video compression network evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--dataset", type=str,default='/bao/wff/media/HEVC_D', help="sequences directory")
    parent_parser.add_argument("--output", type=str,default='../result/exp_01_mse_q3', help="output directory")
    parent_parser.add_argument(
        "-a",
        "--architecture",
        type=str,
        help="model architecture",
        required=False,
    )
    parent_parser.add_argument(
        "-i",
        "--intra-architecture",
        type=str,
        help="model architecture",
    )
    parent_parser.add_argument(
        "-ri",
        "--rate_index",
        nargs="+",
        type=int,
        default=(3,),
    )
    parent_parser.add_argument(
        "-f", "--force", action="store_true", help="overwrite previous runs"
    )
    parent_parser.add_argument("--cuda", action="store_true", help="use cuda")
    parent_parser.add_argument("--half", action="store_true", help="use AMP")
    parent_parser.add_argument(
        "-est",
        "--entropy-estimation",
        action="store_true",
        help="use evaluated entropy estimation (no entropy coding)",
    )
    parent_parser.add_argument(
        "-c",
        "--entropy-coder",
        choices=compressai.available_entropy_coders(),
        default=compressai.available_entropy_coders()[0],
        help="entropy coder (default: %(default)s)",
    )
    parent_parser.add_argument(
        "--keep_binaries",
        action="store_true",
        help="keep bitstream files in output directory",
    )
    parent_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="verbose mode",
    )
    parent_parser.add_argument(
        "-s",
        "--savedir",
        type=str,
        default="",
    )
    parent_parser.add_argument("--save", action="store_true", help="save img")
    parent_parser.add_argument(
        "-m",
        "--metric",
        type=str,
        choices=["mse", "ms-ssim"],
        default="mse",
        help="metric trained against (default: %(default)s)",
    )
    parent_parser.add_argument(
        "--frames",
        type=int,
        default=-1,
    )
    parent_parser.add_argument(
        "--gop",
        type=int,
        default=32,
    )
    subparsers = parser.add_subparsers(help="model source", dest="source")
    subparsers.required = True

    pretrained_parser = subparsers.add_parser("pretrained", parents=[parent_parser])
    pretrained_parser.add_argument(
        "-q",
        "--quality",
        dest="qualities",
        nargs="+",
        type=int,
        default=(1,),
    )

    pretrained_parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    checkpoint_parser = subparsers.add_parser("checkpoint", parents=[parent_parser])
    checkpoint_parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    checkpoint_parser.add_argument('-exp','--experiment',type=str, required=True, help='Experiment name')
    return parser


def main(args: Any = None) -> None:
    if args is None:
        args = sys.argv[1:]
    parser = create_parser()
    args = parser.parse_args(args)

    if not args.source:
        print("Error: missing 'checkpoint' or 'pretrained' source.", file=sys.stderr)
        parser.print_help()
        raise SystemExit(1)
    description = (
        "entropy-estimation" if args.entropy_estimation else args.entropy_coder
    )
    filepaths = collect_videos(args.dataset)
    if len(filepaths) == 0:
        print("Error: no video found in directory.", file=sys.stderr)
        raise SystemExit(1)

    outputdir = args.output
    Path(outputdir).mkdir(parents=True, exist_ok=True)
    time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
    outputdir = Path(outputdir)/os.path.basename(args.dataset)
    outputdir = outputdir/time_stamp
    runs = sorted(args.rate_index)
    opts = (args.architecture,)
    load_func = load_net
    log_fmt = "\rEvaluating {run:s}"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    results = defaultdict(list)

    p_frame_y_q_scales, p_frame_mv_y_q_scales = DSCVC.get_q_scales_from_ckpt(args.experiment)

    print("y_q_scales in inter ckpt: ", end='')
    for q in p_frame_y_q_scales:
        print(f"{q:.3f}, ", end='')
    print()
    print("mv_y_q_scales in inter ckpt: ", end='')
    for q in p_frame_mv_y_q_scales:
        print(f"{q:.3f}, ", end='')
    for run in runs:
        if args.verbose:
            sys.stderr.write(log_fmt.format(*opts, run=run))
            sys.stderr.flush()
        model = load_func(*opts, args.experiment)
        intra_model = IntraModel().to("cuda")
        intra_ckpt_list = ['ckpts/intra_q1.pth.tar', 'ckpts/intra_q2.pth.tar',
                           'ckpts/intra_q3.pth.tar', 'ckpts/intra_q4.pth.tar']
        intra_model =intra_model.from_state_dict(torch.load(intra_ckpt_list[run])).eval()
        ckpt_name = Path(args.experiment).parts[-1]
        trained_net = f"{ckpt_name}-{description}-q{run}"
        print(f"Using trained model {trained_net}", file=sys.stderr)
        if args.cuda and torch.cuda.is_available():
            model = model.to("cuda")
            intra_model = intra_model.to("cuda")
            if args.half:
                model = model.half()
                intra_model = intra_model.half()
        args_dict = vars(args)
        idx = run
        metrics = run_inference(
            filepaths,
            intra_model,
            model,
            idx,
            p_frame_y_q_scales[idx],
            p_frame_mv_y_q_scales[idx],
            outputdir,
            trained_net=trained_net,
            description=description,
            **args_dict,
        )
        results["q"].append(trained_net)
        for k, v in metrics.items():
            results[k].append(v)

    output = {
        "name": f"{args.architecture}-{args.metric}",
        "description": f"Inference ({description})",
        "results": results,
    }

    with (Path(f"{outputdir}/{args.architecture}-{description}-{trained_net}.json")).open("wb") as f:
        f.write(json.dumps(output, indent=2).encode())
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])