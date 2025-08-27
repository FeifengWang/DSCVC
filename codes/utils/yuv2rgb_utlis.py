from typing import Tuple, Union

import torch
import torch.nn.functional as F

from torch import Tensor

YCBCR_WEIGHTS = {
    # Spec: (K_r, K_g, K_b) with K_g = 1 - K_r - K_b
    "ITU-R_BT.709": (0.2126, 0.7152, 0.0722),
    "ITU-R_BT.601": (0.299, 0.587, 0.114)  # RGB到Y的权重
}


def _check_input_tensor(tensor: Tensor) -> None:
    if (
            not isinstance(tensor, Tensor)
            or not tensor.is_floating_point()
            or not len(tensor.size()) in (3, 4)
            or not tensor.size(-3) == 3
    ):
        raise ValueError(
            "Expected a 3D or 4D tensor with shape (Nx3xHxW) or (3xHxW) as input"
        )


def rgb2ycbcr(rgb: Tensor, std="ITU-R_BT.601") -> Tensor:
    """RGB to YCbCr conversion for torch Tensor.
    Using ITU-R BT.709 coefficients.

    Args:
        rgb (torch.Tensor): 3D or 4D floating point RGB tensor

    Returns:
        ycbcr (torch.Tensor): converted tensor
    """
    _check_input_tensor(rgb)

    r, g, b = rgb.chunk(3, -3)
    Kr, Kg, Kb = YCBCR_WEIGHTS[std]
    y = Kr * r + Kg * g + Kb * b
    cb = 0.5 * (b - y) / (1 - Kb) + 0.5
    cr = 0.5 * (r - y) / (1 - Kr) + 0.5

    y = y * (219.0 / 255.0) + 16.0 / 256
    cb = cb * (224.0 / 255.0) + 16.0 / 256
    cr = cr * (224.0 / 255.0) + 16.0 / 256

    ycbcr = torch.cat((y, cb, cr), dim=-3)
    return ycbcr


# def ycbcr2rgb(ycbcr: Tensor) -> Tensor:
#     """YCbCr to RGB conversion for torch Tensor.
#     Using ITU-R BT.709 coefficients.
#
#     Args:
#         ycbcr (torch.Tensor): 3D or 4D floating point RGB tensor
#
#     Returns:
#         rgb (torch.Tensor): converted tensor
#     """
#     _check_input_tensor(ycbcr)
#
#     y, cb, cr = ycbcr.chunk(3, -3)
#     Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
#     r = y + (2 - 2 * Kr) * (cr - 0.5)
#     b = y + (2 - 2 * Kb) * (cb - 0.5)
#     g = (y - Kr * r - Kb * b) / Kg
#
#     r = 1.164 * (y - 16/256.) + 1.793 * (cr - 0.5)
#     g = 1.164 * (y - 16/256.) - 0.534 * (cr - 0.5) - 0.213 * (cb - 0.5)
#     b = 1.164 * (y - 16/256.) + 2.115 * (cb - 0.5)
#     r = r.clamp(0, 1)
#     g = g.clamp(0, 1)
#     b = b.clamp(0, 1)
#
#     rgb = torch.cat((r, g, b), dim=-3)
#     return rgb

def ycbcr2rgb(ycbcr: Tensor, std="ITU-R_BT.601") -> Tensor:
    """YCbCr to RGB conversion for torch Tensor.
    Using ITU-R BT.601 coefficients.

    Args:
        ycbcr (torch.Tensor): 3D or 4D floating point RGB tensor

    Returns:
        rgb (torch.Tensor): converted tensor
    """
    _check_input_tensor(ycbcr)

    y, cb, cr = ycbcr.chunk(3, -3)
    #full range
    y = (y - 16.0 / 256) * (255.0 / 219.0)
    cb = (cb - 16.0 / 256) * (255.0 / 224.0)
    cr = (cr - 16.0 / 256) * (255.0 / 224.0)

    Kr, Kg, Kb = YCBCR_WEIGHTS[std]
    r = y + (2 - 2 * Kr) * (cr - 0.5)
    b = y + (2 - 2 * Kb) * (cb - 0.5)
    g = (y - Kr * r - Kb * b) / Kg

    r = r.clamp(0, 1)
    g = g.clamp(0, 1)
    b = b.clamp(0, 1)

    rgb = torch.cat((r, g, b), dim=-3)
    return rgb


def yuv_444_to_420(
        yuv: Union[Tensor, Tuple[Tensor, Tensor, Tensor]],
        mode: str = "avg_pool",
) -> Tuple[Tensor, Tensor, Tensor]:
    """Convert a 444 tensor to a 420 representation.

    Args:
        yuv (torch.Tensor or (torch.Tensor, torch.Tensor, torch.Tensor)): 444
            input to be downsampled. Takes either a (Nx3xHxW) tensor or a tuple
            of 3 (Nx1xHxW) tensors.
        mode (str): algorithm used for downsampling: ``'avg_pool'``. Default
            ``'avg_pool'``

    Returns:
        (torch.Tensor, torch.Tensor, torch.Tensor): Converted 420
    """
    if mode not in ("avg_pool",):
        raise ValueError(f'Invalid downsampling mode "{mode}".')

    if mode == "avg_pool":
        def _downsample(tensor):
            return F.avg_pool2d(tensor, kernel_size=2, stride=2)

    if isinstance(yuv, torch.Tensor):
        y, u, v = yuv.chunk(3, 1)
    else:
        y, u, v = yuv

    return (y, _downsample(u), _downsample(v))


def yuv_420_to_444(
        yuv: Tuple[Tensor, Tensor, Tensor],
        mode: str = "bilinear",
        return_tuple: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
    """Convert a 420 input to a 444 representation.

    Args:
        yuv (torch.Tensor, torch.Tensor, torch.Tensor): 420 input frames in
            (Nx1xHxW) format
        mode (str): algorithm used for upsampling: ``'bilinear'`` |
            | ``'bilinear'`` | ``'nearest'`` Default ``'bilinear'``
        return_tuple (bool): return input as tuple of tensors instead of a
            concatenated tensor, 3 (Nx1xHxW) tensors instead of one (Nx3xHxW)
            tensor (default: False)

    Returns:
        (torch.Tensor or (torch.Tensor, torch.Tensor, torch.Tensor)): Converted
            444
    """
    if len(yuv) != 3 or any(not isinstance(c, torch.Tensor) for c in yuv):
        raise ValueError("Expected a tuple of 3 torch tensors")

    if mode not in ("bilinear", "bicubic", "nearest"):
        raise ValueError(f'Invalid upsampling mode "{mode}".')

    kwargs = {}
    if mode != "nearest":
        kwargs = {"align_corners": False}

    def _upsample(tensor):
        return F.interpolate(tensor, scale_factor=2, mode=mode, **kwargs)

    y, u, v = yuv
    u, v = _upsample(u), _upsample(v)
    if return_tuple:
        return y, u, v
    return torch.cat((y, u, v), dim=1)


if __name__ == "__main__":
    import os


    def is_yuv_file(filename):
        return any(filename.endswith(extension) for extension in ["yuv"])


    path = '/media/sugon/新加卷/wff/SCC/SCC-SEQ'
    yuv_list = [os.path.join(path, x) for x in os.listdir(path) if is_yuv_file(x)]
    # for yuv_file in sorted(yuv_list):
    #     org_seq = RawVideoSequence.from_file(yuv_file)
