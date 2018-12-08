import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x

'''
Basic convolution block consisting of convolution layer, batch norm and leaky relu
'''
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(BasicConv, self).__init__()
        self.basic_block = nn.Sequential(
                                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding),
                                nn.BatchNorm2d(out_channels),
                                nn.LeakyReLU(0.2, inplace=True),
                            )

    def forward(self, inp):
        out = self.basic_block(inp)
        return out
    

'''
Basic transpose convolution block consisting of transpose convolution layer, batch norm and leaky relu
'''
class BasicConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(BasicConvTranspose, self).__init__()
        self.basic_block = nn.Sequential(
                                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding),
                                nn.BatchNorm2d(out_channels),
                                nn.LeakyReLU(0.2, inplace=True),
                            )
    def forward(self, inp):
        out = self.basic_block(inp)
        return out
    
'''
Encoder network based on DC-GAN architecture
'''
class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=False, vae=False):
        super(Encoder, self).__init__()

        self.normalize = normalize
        self.vae = vae
        if vae:
            out_channels = 2 * out_channels

        self.block1 = BasicConv(in_channels, 64, 4, 2, 1)
        self.block2 = BasicConv(64, 128, 4, 2, 1)
        self.block3 = BasicConv(128, 256, 4, 2, 1)
        self.block4 = BasicConv(256, 512, 4, 2, 1)
        self.block5 = nn.Sequential(
                        nn.Conv2d(512, out_channels, 4, 1, 0),
                        nn.BatchNorm2d(out_channels),
                        nn.Tanh(),
                        )
        
    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
        
    def forward(self, inp):
        out1 = self.block1(inp)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)
        out5 = self.block5(out4)

        if self.normalize:
            out5 = F.normalize(out6, p=2)
            
        if self.vae:
            mu, log_var = torch.split(out5, out5.size(1) // 2, 1)
            out5 = self.reparameterize(mu, log_var)
            return out5, mu, log_var, [out1, out2, out3, out4]
        
        else:
            return out5, [out1, out2, out3, out4]
    
'''
Decoder based on DC-GAN architecture
'''
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, use_skip=False):
        super(Decoder, self).__init__()

        self.use_skip = use_skip
        # if the skip connections are used, then the input at each stage is the
        # concatenation of current feature and feature vector from the encoder
        # hence double the channels, so mul_factor (multiplication factor) is 
        # used to incorporate this effect
        self.mul_factor = 1
        if self.use_skip:
            self.mul_factor = 2

        self.block1 = BasicConvTranspose(in_channels, 512, 4, 1, 0)
        self.block2 = BasicConvTranspose(512*self.mul_factor, 256, 4, 2, 1)
        self.block3 = BasicConvTranspose(256*self.mul_factor, 128, 4, 2, 1)
        self.block4 = BasicConvTranspose(128*self.mul_factor, 64, 4, 2, 1)
        self.block5 = nn.Sequential(
                    nn.ConvTranspose2d(64*self.mul_factor, out_channels, 4, 2, 1),
                    nn.Sigmoid()
                    )
        
    def forward(self, content, skip, pose):
        if pose is not None:
            inp1 = torch.cat([content, pose], dim=1)
        else:
            inp1 = content
        out = self.block1(inp1)
        
        if self.use_skip:
            out = self.block2(torch.cat([out, skip[3]], dim=1))
            out = self.block3(torch.cat([out, skip[2]], dim=1))
            out = self.block4(torch.cat([out, skip[1]], dim=1))
            out = self.block5(torch.cat([out, skip[0]], dim=1))
        else:
            out = self.block2(out)
            out = self.block3(out)
            out = self.block4(out)
            out = self.block5(out)
        return out
    
'''
Discriminator for WGAN
'''
class Discriminator(nn.Module):
    def __init__(self, in_channel, img_shape):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channel,32,3,2,1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(32,32,3,1,1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(32,32,3,2,1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(32,32,3,1,1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(32,32,3,2,1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            Flatten(),
            nn.Linear(8*8*32, 1),
            nn.Hardtanh()
            
        )

    def forward(self, img):
#         img_flat = img.view(img.shape[0], -1)
        validity = self.model(img)
        return validity