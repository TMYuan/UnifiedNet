import torch
import torch.nn as nn
import torch.nn.functional as F
from .pretrain import resnet50

class Encoder(nn.Module):
    """
    This module is used to encode optical flow & images(5 channels)
    into latent space.
    
    ***
    Use Resnet50 to get downsampled 1/8x of original image size.
    ***
    """
    def __init__(self, vae=False):
        super(Encoder, self).__init__()
        resnet = resnet50(True, num_classes=1000)

        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.vae = vae
        if vae:
            self.final_conv = nn.Conv2d(512, 2, kernel_size=1)
        else:
            self.final_conv = nn.Conv2d(512, 1, kernel_size=1)
        
    def forward(self, x):
        # TODO: Concat of flows and images
        x = self.features(x)
        x = self.final_conv(x)
        return x

class Decoder(nn.Module):
    """
    This module is used to decode latent variable & image back to
    optical flow.
    ***
    Use upsample & conv to achieve transposed conv
    ***
    """
    def __init__(self):
        super(Decoder, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 4, kernel_size=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(7, 28, kernel_size=1),
            nn.BatchNorm2d(28),
            nn.Conv2d(28, 28, kernel_size=3, padding=1),
            nn.BatchNorm2d(28),
            nn.Conv2d(28, 7, kernel_size=1),
            nn.BatchNorm2d(7),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(10, 40, kernel_size=1),
            nn.BatchNorm2d(40),
            nn.Conv2d(40, 40, kernel_size=3, padding=1),
            nn.BatchNorm2d(40),
            nn.Conv2d(40, 10, kernel_size=1),
            nn.BatchNorm2d(10),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.final_conv = nn.Conv2d(13, 1, kernel_size=1)
        self.downsample = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)

    def forward(self, x, c):
        c_2 = self.downsample(c)
        c_4 = self.downsample(c_2)
        c_8 = self.downsample(c_4)

        x = self.block1(torch.cat([x, c_8], 1))
        x = self.block2(torch.cat([x, c_4], 1))
        x = self.block3(torch.cat([x, c_2], 1))
        x = self.final_conv(torch.cat([x, c], 1))

        return x

def encoder(**kwargs):
    return Encoder(**kwargs)

def decoder(**kwargs):
    return Decoder(**kwargs)

if __name__ == '__main__':
    model = Encoder(vae=False)
    print(model)