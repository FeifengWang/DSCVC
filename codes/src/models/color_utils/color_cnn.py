import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


from src.models.color_utils.unet import UNet
from src.models.color_utils.dncnn import DnCNN
def tensor_to_PIL(tensor):
    unloader = transforms.ToPILImage()

    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image
class ColorCNN(nn.Module):
    def __init__(self, arch, num_colors=4, soften=1, color_norm=1, color_jitter=0):
        super().__init__()
        self.num_colors = num_colors
        self.soften = soften
        self.color_norm = color_norm
        self.color_jitter = color_jitter
        self.base = UNet(3) if arch == 'unet' else DnCNN(3)
        self.color_mask = nn.Sequential(nn.Conv2d(self.base.out_channel, 256, 1), nn.ReLU(),
                                        nn.Conv2d(256, num_colors, 1, bias=False))
        self.mask_softmax = nn.Softmax2d()

    def forward(self, img, training=True):
        feat = self.base(img)
        m = self.color_mask(feat)
        m = self.mask_softmax(self.soften * m)  # softmax output B,num_colors, H, W
        M = torch.argmax(m, dim=1, keepdim=True)  # argmax color index map  B,1, H, W
        indicator_M = torch.zeros_like(m).scatter(1, M, 1) #B,num_colors, H, W
        if training:
            color_palette = (img.unsqueeze(2) * m.unsqueeze(1)).sum(dim=[3, 4], keepdim=True) / (
                    m.unsqueeze(1).sum(dim=[3, 4], keepdim=True) + 1e-8) / self.color_norm
            jitter_color_palette = color_palette + self.color_jitter * np.random.randn()
            transformed_img = (m.unsqueeze(1) * jitter_color_palette).sum(dim=2)
        else:
            color_palette = (img.unsqueeze(2) * indicator_M.unsqueeze(1)).sum(dim=[3, 4], keepdim=True) / (
                    indicator_M.unsqueeze(1).sum(dim=[3, 4], keepdim=True) + 1e-8)
            transformed_img = (indicator_M.unsqueeze(1) * color_palette).sum(dim=2)

        return transformed_img, m, color_palette


class PaletteCNN(nn.Module):
    def __init__(self, arch='unet', num_colors=4, soften=1, color_norm=1, color_jitter=0):
        super().__init__()
        self.num_colors = num_colors
        self.soften = soften
        self.color_norm = color_norm
        self.color_jitter = color_jitter
        self.base = UNet(3) if arch == 'unet' else DnCNN(3)
        self.color_mask = nn.Sequential(nn.Conv2d(self.base.out_channel, 256, 1), nn.ReLU(),
                                        nn.Conv2d(256, num_colors, 1, bias=False))
        self.mask_softmax = nn.Softmax2d()

    def forward(self, img, training=True):
        semantic_feature = self.base(img)
        m = self.color_mask(semantic_feature)
        m = self.mask_softmax(self.soften * m)  # softmax output B,num_colors, H, W
        # color_palette = (img.unsqueeze(2) * m.unsqueeze(1)).sum(dim=[3, 4], keepdim=True) / (
        #         m.unsqueeze(1).sum(dim=[3, 4], keepdim=True) + 1e-8) / self.color_norm
        # color_palette = color_palette
        # transformed_img = (m.unsqueeze(1) * color_palette).sum(dim=2)
        # return m, color_palette
        if training:
            color_palette = (img.unsqueeze(2) * m.unsqueeze(1)).sum(dim=[3, 4], keepdim=True) / (
                    m.unsqueeze(1).sum(dim=[3, 4], keepdim=True) + 1e-8) / self.color_norm
            color_palette = color_palette + self.color_jitter * np.random.randn()
            # transformed_img = (m.unsqueeze(1) * color_palette).sum(dim=2)
            return m, color_palette
        else:
            M = torch.argmax(m, dim=1, keepdim=True)  # argmax color index map  B,1, H, W
            indicator_M = torch.zeros_like(m).scatter(1, M, 1)  # B,num_colors, H, W
            color_palette = (img.unsqueeze(2) * indicator_M.unsqueeze(1)).sum(dim=[3, 4], keepdim=True) / (
                    indicator_M.unsqueeze(1).sum(dim=[3, 4], keepdim=True) + 1e-8)
            # transformed_img = (indicator_M.unsqueeze(1) * color_palette).sum(dim=2)
            return indicator_M, color_palette


def image_loader(image_name,device):
    loader = transforms.Compose([
        transforms.ToTensor()])
    image = Image.open(image_name).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)
def tensor_to_PIL(tensor):
    unloader = transforms.ToPILImage()

    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image
def one_test():
    device = 'cuda:0'
    img_tensor = image_loader('../imgs/0_patch_47.jpg',device=device)
    model = ColorCNN(3, 4).to(device)
    transformed_tensor, m, color_palette = model(img_tensor)
    print('mask:',m)
    print('plt:',color_palette)
    transformed_img =tensor_to_PIL(transformed_tensor)
    transformed_img.save('0_patch_47_color_cnn.jpg')
    pass

# one_test()
