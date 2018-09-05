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
        resnet = resnet50(True, num_classes=1000, input_channel=2)

        self.features = nn.Sequential(*list(resnet.children())[:-3])
        self.vae = vae
        if vae:
            self.final_conv = nn.Conv2d(1024, 2, kernel_size=3, padding=1)
        else:
            self.final_conv = nn.Conv2d(1024, 1, kernel_size=3, padding=1)
        
    def forward(self, x, c):
        # TODO: Concat of flows and images
        x = self.features(torch.cat([x, c], 1))
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
            nn.Conv2d(2, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=1),
            nn.BatchNorm2d(32),
#             nn.Tanh(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(33, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
#             nn.Tanh(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(65, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.BatchNorm2d(128),
#             nn.Tanh(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(129, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
#             nn.Tanh(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(257, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, c):
        # c_{} means down-sampling factor
        c_1 = c
        c_2 = F.avg_pool2d(c, 2)
        c_4 = F.avg_pool2d(c, 4)
        c_8 = F.avg_pool2d(c, 8)
        c_16 = F.avg_pool2d(c, 16)
#         c_1, c_2, c_4, c_8, c_16 = c

        x = self.block1(torch.cat([x, c_16], 1))
        x = self.block2(torch.cat([x, c_8], 1))
        x = self.block3(torch.cat([x, c_4], 1))
        x = self.block4(torch.cat([x, c_2], 1))
        x = self.final_conv(torch.cat([x, c_1], 1))

        return x
    
class ImageEncoder(nn.Module):
    """
    This module is used to extract features of images from resnet 50
    
    ***
    Use Resnet50 to get downsampled 1/16x of original image size.
    224*224 -> 14*14
    ***
    """
    def __init__(self):
        super(ImageEncoder, self).__init__()
        resnet = resnet50(True, num_classes=1000, input_channel=1)

#         self.features = nn.Sequential(*list(resnet.children())[:-3])
        self.f1 = nn.Sequential(*list(resnet.children())[:3])
        self.f2 = nn.Sequential(*list(resnet.children())[3:5])
        self.f3 = nn.Sequential(*list(resnet.children())[5])
        self.f4 = nn.Sequential(*list(resnet.children())[6])
        
    def forward(self, x):
        c_2 = self.f1(torch.cat([x], 1))
        c_4 = self.f2(c_2)
        c_8 = self.f3(c_4)
        c_16 = self.f4(c_8)
        return x, c_2, c_4, c_8, c_16

    
class MNISTEncoder(nn.Module):
    """
    This module is encoder definition for Moving MNIST
    
    ***
    Input image size: 64*64
    ***
    """
    def __init__(self, vae=False):
        super(MNISTEncoder, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.vae = vae
        if vae:
            self.final_conv = nn.Conv2d(256, 2, kernel_size=3, padding=1)
        else:
            self.final_conv = nn.Conv2d(256, 1, kernel_size=3, padding=1)
        
    def forward(self, img2, img1):
        # TODO: Concat of flows and images
        x = self.features(torch.cat([img2, img1], 1))
        x = self.final_conv(x)
        return x
    
class MNISTDecoder(nn.Module):
    """
    This module is decoder definition for Moving MNIST
    
    ***
    Input: latent variable and img_1
    ***
    """
    def __init__(self):
        super(MNISTDecoder, self).__init__()
#         self.block1 = nn.Sequential(
#             nn.Conv2d(2, 256, kernel_size=3, padding=1),
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         )
#         self.block2 = nn.Sequential(
#             nn.Conv2d(257, 128, kernel_size=3, padding=1),
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         )
#         self.block3 = nn.Sequential(
#             nn.Conv2d(129, 64, kernel_size=3, padding=1),
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         )
#         self.block4 = nn.Sequential(
#             nn.Conv2d(65, 32, kernel_size=3, padding=1),
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         )
#         self.final = nn.Sequential(
#             nn.Conv2d(33, 1, kernel_size=3, padding=1),
#             nn.Sigmoid()
#         )
        self.block1 = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 4, kernel_size=1),
            nn.BatchNorm2d(4),
#             nn.Tanh(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(5, 28, kernel_size=1),
            nn.BatchNorm2d(28),
            nn.Conv2d(28, 28, kernel_size=3, padding=1),
            nn.BatchNorm2d(28),
            nn.Conv2d(28, 7, kernel_size=1),
            nn.BatchNorm2d(7),
#             nn.Tanh(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(8, 40, kernel_size=1),
            nn.BatchNorm2d(40),
            nn.Conv2d(40, 40, kernel_size=3, padding=1),
            nn.BatchNorm2d(40),
            nn.Conv2d(40, 10, kernel_size=1),
            nn.BatchNorm2d(10),
#             nn.Tanh(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(11, 52, kernel_size=1),
            nn.BatchNorm2d(52),
            nn.Conv2d(52, 52, kernel_size=3, padding=1),
            nn.BatchNorm2d(52),
            nn.Conv2d(52, 13, kernel_size=1),
            nn.BatchNorm2d(13),
#             nn.Tanh(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(14, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, z, img1):
        # c_{} means down-sampling factor
        c_2 = F.avg_pool2d(img1, 2)
        c_4 = F.avg_pool2d(img1, 4)
        c_8 = F.avg_pool2d(img1, 8)
        c_16 = F.avg_pool2d(img1, 16)
        
        x = self.block1(torch.cat([z, c_16], 1))
        x = self.block2(torch.cat([x, c_8], 1))
        x = self.block3(torch.cat([x, c_4], 1))
        x = self.block4(torch.cat([x, c_2], 1))
        x = self.final_conv(torch.cat([x, img1], 1))
        return x
    

def encoder(**kwargs):
    return Encoder(**kwargs)

def decoder(**kwargs):
    return Decoder(**kwargs)

def image_encoder(**kwargs):
    return ImageEncoder(**kwargs)