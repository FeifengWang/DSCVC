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
import logging
import math
import os
import random
import shutil
import sys

from collections import defaultdict
from typing import List, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from pytorch_msssim import ms_ssim

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms import ToPILImage

from dataset_utils.scv_dataset import SCVFolder

# from compressai.datasets import VideoFolder
# from compressai.utils.bench.codecs import compute_metrics
from compressai.zoo import video_models

from src.models.intra_model_scc import IntraModel
from src.models.video_model import DSCVC
from utils import util
def torch2img(x: torch.Tensor) -> Image.Image:
    return ToPILImage()(x.clamp_(0, 1).squeeze())
def compute_metrics(
    a: Union[np.array, Image.Image],
    b: Union[np.array, Image.Image],
    max_val: float = 255.0,
) -> Tuple[float, float]:
    """Returns PSNR and MS-SSIM between images `a` and `b`. """
    if isinstance(a, Image.Image):
        a = np.asarray(a)
    if isinstance(b, Image.Image):
        b = np.asarray(b)

    a = torch.from_numpy(a.copy()).float().unsqueeze(0)
    if a.size(3) == 3:
        a = a.permute(0, 3, 1, 2)
    b = torch.from_numpy(b.copy()).float().unsqueeze(0)
    if b.size(3) == 3:
        b = b.permute(0, 3, 1, 2)

    mse = torch.mean((a - b) ** 2).item()
    if mse==0.:
        return 100.,1.
    p = 20 * np.log10(max_val) - 10 * np.log10(mse)
    m = ms_ssim(a, b, data_range=max_val).item()
    return p, m
def collect_likelihoods_list(likelihoods_list, num_pixels: int):
    bpp_info_dict = defaultdict(int)
    bpp_loss = 0

    for i, frame_likelihoods in enumerate(likelihoods_list):
        frame_bpp = 0
        for label, likelihoods in frame_likelihoods.items():
            label_bpp = 0
            for field, v in likelihoods.items():
                bpp = torch.log(v).sum(dim=(1, 2, 3)) / (-math.log(2) * num_pixels)

                bpp_loss += bpp
                frame_bpp += bpp
                label_bpp += bpp

                bpp_info_dict[f"bpp_loss.{label}"] += bpp.sum()
                bpp_info_dict[f"bpp_loss.{label}.{i}.{field}"] = bpp.sum()
            bpp_info_dict[f"bpp_loss.{label}.{i}"] = label_bpp.sum()
        bpp_info_dict[f"bpp_loss.{i}"] = frame_bpp.sum()
    return bpp_loss, bpp_info_dict



class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)
def configure_optimizers(net, args):
    """Separate parameters for the main optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )

    return optimizer


def train_one_epoch(
    model, train_dataloader, optimizer, epoch, clip_max_norm, logger_train, logger_total, tb_logger, current_step, args
):
    model.train()
    device = next(model.parameters()).device


    for i, batch in enumerate(train_dataloader):
        d = [frames.to(device) for frames in batch]
        if args.rate_adaptive:
            q_scale_index = np.random.randint(0, len(args.lmbda_list))
        else:
            q_scale_index = args.q_scale_index
        optimizer.zero_grad()

        out_net = model(d, q_scale_index)



        mse_loss = out_net['mse_loss'].mean()
        warploss = out_net['warploss'].mean()

        bpp_y = out_net["bpp_y"].mean()
        bpp_z = out_net["bpp_z"].mean()
        bpp_mv_y = out_net["bpp_mv_y"].mean()
        bpp_mv_z = out_net["bpp_mv_z"].mean()
        bpp = out_net["bpp"].mean()
        if args.finetune:
            mse_loss = out_net['weighted_mse_loss'].mean()
        distortion = mse_loss + warploss
        bpp_loss =bpp_mv_y + bpp_mv_z + bpp_y + bpp_z

        lmbda = args.lmbda_list[q_scale_index]
        rd_loss = lmbda * distortion + bpp_loss


        # tc = datetime.datetime.now()
        rd_loss.backward()

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()


        current_step += 1
        if current_step % 100 == 0:
            tb_logger.add_scalar('{}'.format('[train]: loss'), rd_loss.item(), current_step)
            tb_logger.add_scalar('{}'.format('[train]: bpp_loss'), bpp_loss.item(), current_step)
            tb_logger.add_scalar('{}'.format('[train]: bpp_motion'), (bpp_mv_y+bpp_mv_z).item(), current_step)
            tb_logger.add_scalar('{}'.format('[train]: bpp_residual'), (bpp_y+bpp_z).item(), current_step)
            tb_logger.add_scalar('{}'.format('[train]: mse_loss'), mse_loss.item(), current_step)
            tb_logger.add_scalar('{}'.format('[train]: warp_loss'), warploss.item(), current_step)
            # if out_criterion["mse_loss"] is not None:
            #     tb_logger.add_scalar('{}'.format('[train]: mse_loss'), mse_loss.item(), current_step)
            #     tb_logger.add_scalar('{}'.format('[train]: warp_loss'), warploss.item(), current_step)
            # if out_criterion["ms_ssim_loss"] is not None:
            #     tb_logger.add_scalar('{}'.format('[train]: ms_ssim_loss'), out_criterion["ms_ssim_loss"].item(), current_step)
        if current_step % 50000==0:
            if args.save:
                state = {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "loss": 1e10,
                    # "optimizer": optimizer.state_dict(),
                    # "lr_scheduler": lr_scheduler.state_dict(),
                }
                filename = os.path.join('../experiments', args.experiment, 'checkpoints',
                                        "checkpoint_epoch_{:0>3d}_iter_{:}.pth.tar".format(epoch + 1,current_step))
                # save_checkpoint(state,is_best,filename)
                torch.save(state, filename)
        if i % 100 == 0:
            logger_train.info(
                f"Train epoch {epoch}: ["
                # f"{i*len(d):5d}/{len(train_dataloader.dataset)}"
                f"{i:5d}/{len(train_dataloader)}"
                f" ({100. * i / len(train_dataloader):.0f}%)] "
                f'Loss: {rd_loss.item():.4f} | '
                f'MSE loss: {mse_loss.item():.4f} | '
                f'WARP loss: {warploss.item():.4f} | '
                f'Bpp loss: {bpp_loss.item():.4f} | '

            )
            logger_total.info(
                f"Train epoch {epoch}: ["
                # f"{i*len(d):5d}/{len(train_dataloader.dataset)}"
                f"{i:5d}/{len(train_dataloader)}"
                f" ({100. * i / len(train_dataloader):.0f}%)] "
                f'Loss: {rd_loss.item():.4f} | '
                f'MSE loss: {mse_loss.item():.4f} | '
                f'WARP loss: {warploss.item():.4f} | '
                f'Bpp loss: {bpp_loss.item():.4f} | '

            )

    return current_step

def for_test_epoch(epoch, test_dataloader, model, save_dir, logger_val,logger_total, tb_logger, args):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()

    bpp_loss = AverageMeter()
    bpp_motion = AverageMeter()
    bpp_residual = AverageMeter()
    mse_loss = AverageMeter()

    ms_ssim_loss = AverageMeter()
    psnr = AverageMeter()
    wpsnr = AverageMeter()
    ms_ssim = AverageMeter()
    with torch.no_grad():
        for idx, batch in enumerate(test_dataloader):
            d = [frames.to(device) for frames in batch]

            if args.rate_adaptive:
                q_scale_index = np.random.randint(0, len(args.lmbda_list))
            else:
                q_scale_index = args.q_scale_index
            out_net = model(d, q_scale_index)

            # out_criterion = criterion(out_net, d[1:])


            bpp_loss.update(out_net["bpp"].mean())
            bpp_motion.update(out_net["bpp_mv_y"].mean()+out_net["bpp_mv_z"].mean())
            bpp_residual.update(out_net["bpp_y"].mean()+out_net["bpp_z"].mean())
            loss.update(args.lmbda_list[q_scale_index] * out_net['mse_loss'].mean() + out_net["bpp"].mean())
            mse_loss.update(out_net['mse_loss'].mean())

            # if out_criterion["mse_loss"] is not None:
            #     mse_loss.update(out_criterion["mse_loss"])
            # if out_criterion["ms_ssim_loss"] is not None:
            #     ms_ssim_loss.update(out_criterion["ms_ssim_loss"])
            for i in range(d.__len__()-1):
                for j in range(d[i].shape[0]):
                    rec = torch2img(out_net['x_hat'][i][j])
                    warp = torch2img(out_net['x_warp'][i][j])
                    img = torch2img(d[i+1][j])
                    p, m = compute_metrics(rec, img)
                    w_p,_ = compute_metrics(warp, img)
                    psnr.update(p)
                    wpsnr.update(w_p)
                    ms_ssim.update(m)
                    if idx % 2 == 1:
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                        rec.save(os.path.join(save_dir, 'b%03d' % j+'_f%03d_rec.png' % i))
                        img.save(os.path.join(save_dir, 'b%03d' % j+'_f%03d_gt.png' % i))
                        warp.save(os.path.join(save_dir, 'b%03d' % j+'_f%03d_warp.png' % i))



    tb_logger.add_scalar('{}'.format('[val]: loss'), loss.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: bpp_loss'), bpp_loss.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: bpp_motion'), bpp_motion.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: bpp_residual'), bpp_residual.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: psnr'), psnr.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: warp_psnr'), wpsnr.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: ms-ssim'), ms_ssim.avg, epoch + 1)

    logger_val.info(
        f"Test epoch {epoch}: Average losses: "
        f"Loss: {loss.avg:.4f} | "
        f"MSE loss: {mse_loss.avg:.4f} | "
        f"Bpp loss: {bpp_loss.avg:.4f} | "
        f"PSNR: {psnr.avg:.6f} | "
        f"WPSNR: {wpsnr.avg:.6f} | "
        f"MS-SSIM: {ms_ssim.avg:.6f}"
    )
    logger_total.info(
        f"Test epoch {epoch}: Average losses: "
        f"Loss: {loss.avg:.4f} | "
        f"MSE loss: {mse_loss.avg:.4f} | "
        f"Bpp loss: {bpp_loss.avg:.4f} | "
        f"PSNR: {psnr.avg:.6f} | "
        f"WPSNR: {wpsnr.avg:.6f} | "
        f"MS-SSIM: {ms_ssim.avg:.6f}"
    )
    tb_logger.add_scalar('{}'.format('[val]: mse_loss'), mse_loss.avg, epoch + 1)
    # print(
    #     f"Test epoch {epoch}: Average losses:"
    #     f"\tLoss: {loss.avg:.3f} |"
    #     f"\tMSE loss: {mse_loss.avg:.3f} |"
    #     f"\tBpp loss: {bpp_loss.avg:.2f} |"
    # )

    return loss.avg


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "checkpoint_best_loss.pth.tar")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")

    parser.add_argument(
        "-exp", "--experiment", type=str, required=True, help="Experiment name"
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-dt", "--dataset_type", type=str, default='lscvd',
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=80,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=0,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "-mf",
        "--multi-frames",
        type=int,
        default=2,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument("--dataset_part", action="store_true", help="Using part of dataset")
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1e-2,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument('--lmbda_list',nargs="+",type=int,  default=[16, 32, 64, 128])
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=4,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "-qi",
        "--q_scale_index",
        type=int,
        default=3,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--gpu_id", default='6,7', help="GPU ID")
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("--rate_adaptive", action="store_true", help="Train rate adaptive")
    parser.add_argument("-f","--finetune", action="store_true", help="Use finetune")
    parser.add_argument("--reset_epoch", action="store_true", help="Reset Epoch Count")
    parser.add_argument("--start_epoch", default=0, help="Reset start epoch")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("-c","--checkpoint", type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args

def seed_torch(seed=4096):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
def main(argv):
    args = parse_args(argv)

    if args.seed is not None:
        seed_torch(args.seed)

    # Warning, the order of the transform composition should be kept.
    train_transforms = transforms.Compose(
        [transforms.ToTensor(),transforms.RandomCrop(args.patch_size)]
    )

    test_transforms = transforms.Compose(
        [transforms.ToTensor(),transforms.CenterCrop(args.patch_size)]
    )

    train_dataset = SCVFolder(
        args.dataset,
        args.multi_frames,
        args.dataset_part,
        rnd_interval=False,
        rnd_temp_order=True,
        split="train",
        transform=train_transforms,
    )
    test_dataset = SCVFolder(
        args.dataset,
        args.multi_frames,
        args.dataset_part,
        rnd_interval=False,
        rnd_temp_order=False,
        split="valid",
        transform=test_transforms,
    )
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    # net = video_models[args.model](quality=3)
    net = DSCVC()
    net = net.to(device)
    optimizer = configure_optimizers(net, args)

    # if args.cuda and torch.cuda.device_count() > 1:
    #     net = CustomDataParallel(net)

    if args.multi_frames > 5:
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150, 170], gamma=0.1)
        if args.multi_frames > 6:
            # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[6, 8], gamma=0.1)
            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,15], gamma=0.1)
        if args.finetune:
            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15], gamma=0.1)
    else:
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[600, 700], gamma=0.1)
    # create log file
    if not os.path.exists(os.path.join('../experiments', args.experiment)):
        os.makedirs(os.path.join('../experiments', args.experiment))
    util.setup_logger('train', os.path.join('../experiments', args.experiment), 'train_' + args.experiment, level=logging.INFO,
                        screen=True, tofile=True)
    util.setup_logger('val', os.path.join('../experiments', args.experiment), 'val_' + args.experiment, level=logging.INFO,
                        screen=True, tofile=True)
    util.setup_logger('total', os.path.join('../experiments', args.experiment), 'total_' + args.experiment, level=logging.INFO,
                        screen=False, tofile=True)
    logger_train = logging.getLogger('train')
    logger_val = logging.getLogger('val')
    logger_total = logging.getLogger('total')
    method = os.path.split(os.path.split(os.getcwd())[-2])[-1]
    tb_logger = SummaryWriter(log_dir='../tb_logger/' + method +'_'+args.experiment)
    logger_train.info(args)
    logger_total.info(args)
    os.makedirs(os.path.join('../experiments', args.experiment, 'checkpoints'), exist_ok=True)
    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if "epoch" in checkpoint:
            last_epoch = checkpoint["epoch"] + 1
            start_epoch = checkpoint["epoch"]
        else:
            last_epoch = 1
            start_epoch = 0
        if "state_dict" in checkpoint:
            # net.load_state_dict(checkpoint["state_dict"],strict=True)

            state_dict = checkpoint.get("state_dict", checkpoint)
            net.load_state_dict(state_dict)

        else:
            net.load_state_dict(checkpoint,strict=True)
        # if "optimizer" in checkpoint:
        #     optimizer.load_state_dict(checkpoint["optimizer"])
        # if "lr_scheduler" in checkpoint:
        #     lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        #optimizer.load_state_dict(checkpoint["optimizer"])
        #lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        current_step = start_epoch * math.ceil(len(train_dataloader.dataset) / args.batch_size)
        best_loss = 1e10 #checkpoint['loss'] if checkpoint['loss'] is not None else 1e10
        if args.reset_epoch:
            start_epoch = args.start_epoch
            best_loss = 1e10
            current_step = 0
    else:
        start_epoch = 0
        best_loss = 1e10
        current_step = 0

    # best_loss = float("inf")
    for epoch in range(start_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        logger_train.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        logger_total.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        current_step = train_one_epoch(
            net,
            train_dataloader,
            optimizer,
            epoch,
            args.clip_max_norm,
            logger_train,logger_total,
            tb_logger,
            current_step,
            args
        )
        save_dir = os.path.join('../experiments', args.experiment, 'val_images', '%03d' % (epoch + 1))
        loss = for_test_epoch(epoch, test_dataloader, net, save_dir, logger_val,logger_total, tb_logger,args)
        # lr_scheduler.step(loss)
        lr_scheduler.step()
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        # if args.save:
        #     save_checkpoint(
        #         {
        #             "epoch": epoch,
        #             "state_dict": net.state_dict(),
        #             "loss": loss,
        #             "optimizer": optimizer.state_dict(),
        #             "lr_scheduler": lr_scheduler.state_dict(),
        #         },
        #         is_best,
        #     )
        #     if is_best:
        #         logger_val.info('best checkpoint saved.')
        if args.save:
            state = {
                    "epoch": epoch + 1,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    # "optimizer": optimizer.state_dict(),
                    # "lr_scheduler": lr_scheduler.state_dict(),
                }
            if epoch == args.epochs-1:
                state = {
                    "epoch": epoch + 1,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                }
            filename = os.path.join('../experiments', args.experiment, 'checkpoints', "checkpoint_%03d.pth.tar" % (epoch + 1))
            # save_checkpoint(state,is_best,filename)
            if (epoch + 1) % 10 ==0:
                torch.save(state, filename)

            if is_best:
                dest_filename = filename.replace(filename.split('/')[-1], "checkpoint_best_loss.pth.tar")
                torch.save(state, dest_filename)
                logger_val.info('best checkpoint  saved: Epoch{}'.format(epoch))
                logger_total.info('best checkpoint  saved: Epoch{}'.format(epoch))

if __name__ == "__main__":
    main(sys.argv[1:])
