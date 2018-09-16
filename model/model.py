import torch
import torch.nn as nn
import torch.nn.functional as F
from .pretrain import resnet50, resnet18

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
        resnet = resnet18(True, num_classes=1000, input_channel=2)

        self.features = nn.Sequential(*list(resnet.children())[:-4])
        self.vae = vae
        if vae:
            self.final_conv = nn.Conv2d(1024, 2, kernel_size=3, padding=1)
        else:
            self.final_conv = nn.Conv2d(128, 32, kernel_size=3, padding=1)

    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, c):
        # TODO: Concat of flows and images
        x = self.features(torch.cat([x, c], 1))
        x = self.final_conv(x)
        if self.vae:
            mu, log_var = torch.split(x, 1, 1)
            z = self.reparameterize(mu, log_var)
            return z, mu, log_var
        else:
            return x

class Decoder(nn.Module):
    """
    This module is used to decode latent variable & image back to
    optical flow.
    ***
    Use upsample & conv to achieve transposed conv
    ***
    """
    def __init__(self, f_extracter=True):
        super(Decoder, self).__init__()
        self.img_encoder = image_encoder()
        self.block1 = self._make_block(1025, 512)
        self.block2 = self._make_block(1024, 256)
        self.block3 = self._make_block(512, 64)
        self.block4 = self._make_block(128, 32)
        self.final_conv = nn.Sequential(
            nn.Conv2d(33, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def _make_block(self, in_channel, out_channel, scale=2, bn_first=False):
        block = []
        if bn_first:
            block += [nn.BatchNorm2d(in_channel)]
        block += [
            nn.Conv2d(in_channel, out_channel, kernel_size=1),
            nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, kernel_size=1),
            nn.BatchNorm2d(out_channel),
            nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)
        ]
        return nn.Sequential(*block)

    def forward(self, x, c):
        # c_{} means down-sampling factor
#         c_1 = c
#         c_2 = F.avg_pool2d(c, 2)
#         c_4 = F.avg_pool2d(c, 4)
#         c_8 = F.avg_pool2d(c, 8)
#         c_16 = F.avg_pool2d(c, 16)
        c_16, c_8, c_4, c_2, c_1 = self.img_encoder(c)

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
        resnet = resnet18(True, num_classes=1000, input_channel=1)

        self.f1 = nn.Sequential(*list(resnet.children())[:3])
        self.f2 = nn.Sequential(*list(resnet.children())[3:5])
        self.f3 = nn.Sequential(*list(resnet.children())[5])
#         self.f4 = nn.Sequential(*list(resnet.children())[6])
        
    def forward(self, x):
        c_2 = self.f1(x)
        c_4 = self.f2(c_2)
        c_8 = self.f3(c_4)
#         c_16 = self.f4(c_8)
        return c_8, c_4, c_2, x

    
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
    def __init__(self, f_extracter=False):
        super(MNISTDecoder, self).__init__()
        self.f_extracter = f_extracter
        self.img_encoder = image_encoder()
        if f_extracter:
            self.model = [
                self._make_block(160, 64),
                self._make_block(128, 64),
                self._make_block(128, 32)
#                 self._make_block(128, 32)
            ]
        else:
            self.model = [
                self._make_block(2, 512),
                self._make_block(513, 256),
                self._make_block(257, 64),
                self._make_block(65, 32)
            ]
        self.model += [nn.Sequential(
            nn.Conv2d(33, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )]
        self.model = nn.ModuleList(self.model)
        
    def _make_block(self, in_channel, out_channel, scale=2, bn_first=False):
        block = []
        if bn_first:
            block += [nn.BatchNorm2d(in_channel)]
        block += [
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, kernel_size=1),
            nn.BatchNorm2d(out_channel),
            nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)
        ]
        return nn.Sequential(*block)

    def forward(self, x, c):
        # create c_list based on "f_extracter"
        # True: feature map from img_encoder, False: c with different size
        # c_{} means down-sampling factor
        c_list = []
        if self.f_extracter:
            c_list = list(self.img_encoder(c))
        else:
            for i in reversed(range(5)):
                c_list.append(F.avg_pool2d(c, 2 ** i))

        for c_i, m_i in zip(c_list, self.model):
            x = m_i(torch.cat([x, c_i], 1))
        return x
    
    

def encoder(**kwargs):
    return Encoder(**kwargs)

def decoder(**kwargs):
    return MNISTDecoder(**kwargs)

def image_encoder(**kwargs):
    return ImageEncoder(**kwargs)