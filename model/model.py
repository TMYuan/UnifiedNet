import torch
import torch.nn as nn
import torch.nn.functional as F
from .pretrain import resnet50

class Encoder(nn.Module):
    """
    This module is used to encode optical flow & images(5 channels)
    into latent space.
    
    ***
    Use Resnet50 to get downsampled 1/16x of original image size.
    224*224 -> 14*14
    ***
    """
    def __init__(self, vae=False):
        super(Encoder, self).__init__()
        resnet = resnet50(True, num_classes=1000)

#         self.features = nn.Sequential(*list(resnet.children())[:-3])
        self.f1 = nn.Sequential(*list(resnet.children())[:3])
        self.f2 = nn.Sequential(*list(resnet.children())[3:5])
        self.f3 = nn.Sequential(*list(resnet.children())[5])
        self.f4 = nn.Sequential(*list(resnet.children())[6])
        self.vae = vae
        if vae:
            self.final_conv = nn.Conv2d(1024, 2, kernel_size=3, padding=1)
        else:
            self.final_conv = nn.Conv2d(1024, 1, kernel_size=3, padding=1)
        
    def forward(self, x, c):
        # TODO: Concat of flows and images
        c_2 = self.f1(torch.cat([x, c], 1))
        c_4 = self.f2(c_2)
        c_8 = self.f3(c_4)
        c_16 = self.f4(c_8)
        
        z = self.final_conv(c_16)
        return z, c_2, c_4, c_8

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
            nn.Conv2d(1, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.BatchNorm2d(512),
#             nn.Tanh(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
#             nn.Tanh(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 64, kernel_size=1),
            nn.BatchNorm2d(64),
#             nn.Tanh(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.BatchNorm2d(32),
#             nn.Tanh(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(35, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 2, kernel_size=1)
#             nn.Tanh()
        )

    def forward(self, x, c, c_2, c_4, c_8):
        # c_{} means down-sampling factor
#         c_2 = F.avg_pool2d(c, 2)
#         c_4 = F.avg_pool2d(c_2, 2)
#         c_8 = F.avg_pool2d(c_4, 2)
#         c_16 = F.avg_pool2d(c_8, 2)

        x = self.block1(x)
        x = self.block2(torch.cat([x, c_8], 1))
        x = self.block3(torch.cat([x, c_4], 1))
        x = self.block4(torch.cat([x, c_2], 1))
        x = self.final_conv(torch.cat([x, c], 1))

        return x

def encoder(**kwargs):
    return Encoder(**kwargs)

def decoder(**kwargs):
    return Decoder(**kwargs)

if __name__ == '__main__':
    model = Decoder()
    print(model)