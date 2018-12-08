from tqdm import tqdm
from model import drnet
from util import datasets, plot, kth
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
from util import utils 
from model.vgg import Vgg16
from loss import EdgeLoss
import pickle
import numpy as np
import random
import torch
import torch.optim as optim
import os
import time
import visdom
import torch.nn.functional as F
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
parser.add_argument('--beta1', default=0.5, type=float, help='momentum term for adam')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--log_dir', default='logs', help='base directory to save logs')
parser.add_argument('--data_root', default='/data/dennis/data_uni', help='root directory for data')
parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--epoch_size', type=int, default=600, help='epoch size')
parser.add_argument('--content_dim', type=int, default=128, help='size of the content vector')
parser.add_argument('--motion_dim', type=int, default=128, help='size of the pose vector')
parser.add_argument('--image_width', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--channels', default=3, type=int)
parser.add_argument('--data_threads', type=int, default=5, help='number of parallel data loading threads')
parser.add_argument('--data_type', default='drnet', help='speed up data loading for drnet training')
parser.add_argument('--dataset', default='kth', help='dataset to train with')
parser.add_argument('--max_step', type=int, default=20, help='maximum distance between frames')
parser.add_argument('--device', type=int, default=0, help='GPU number')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--kl_weight', default=1e-4, type=float, help='weight for KLD loss')

opt = parser.parse_args()
opt.vis_name = opt.log_dir
opt.log_dir = '%s/%s%dx%d' % (opt.log_dir, opt.dataset, opt.image_width, opt.image_width)
opt.device =  torch.device("cuda:%d" % (opt.device) if torch.cuda.is_available() else "cpu")

os.makedirs('%s' % opt.log_dir, exist_ok=True)


print(opt)

print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)

## Setup dataset
trans = transforms.Compose([
            transforms.ToTensor(),
        ])
kth_dataset = kth.KTH(train=True, data_root=opt.data_root, seq_len=opt.max_step, image_size=opt.image_width, transforms=trans)
train_loader = DataLoader(dataset=kth_dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True, num_workers=opt.data_threads, pin_memory=True)
test_dataset = kth.KTH(train=False, data_root=opt.data_root, seq_len=20, image_size=opt.image_width, transforms=trans)

def get_training_batch():
    while True:
        for sequence in train_loader:
            yield sequence
train_generator = get_training_batch()

## Network structure
netEC = drnet.Encoder(opt.channels, opt.content_dim).to(opt.device)
netEM = drnet.Encoder(2*opt.channels, opt.motion_dim, vae=True).to(opt.device)
netD = drnet.Decoder(opt.content_dim + opt.motion_dim, opt.channels, use_skip=True).to(opt.device)
netI = drnet.Decoder(opt.content_dim, opt.channels).to(opt.device)
utils.init_weights(netEC)
utils.init_weights(netEM)
utils.init_weights(netD)

## Optimizer
optimizerEC = optim.Adam(netEC.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerEM = optim.Adam(netEM.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerI = optim.Adam(netI.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

## Loss function
# vgg = Vgg16(requires_grad=False).to(opt.device)
# def perceptual_loss(pred, gt):
#     gt_n = utils.normalize_batch(gt)
#     pred_n = utils.normalize_batch(pred)

#     f_gt = vgg(gt_n)
#     f_pred = vgg(pred_n)
    
#     loss = F.l1_loss(f_pred.relu4_3, f_gt.relu4_3) +\
#         0.5 * F.l1_loss(f_pred.relu3_3, f_gt.relu3_3)+\
#         0.25 * F.l1_loss(f_pred.relu2_2, f_gt.relu2_2)+\
#         0.125 * F.l1_loss(f_pred.relu1_2, f_gt.relu1_2)
#     return loss

## Training functions
def train(img_1, img_2):
    optimizerEC.zero_grad()
    optimizerEM.zero_grad()
    optimizerD.zero_grad()
    optimizerI.zero_grad()
    
    with torch.set_grad_enabled(True):
        vec_c, skip = netEC(img_1)
        if netEM.vae:
            vec_m, mu, log_var, _ = netEM(torch.cat([img_1, img_2], dim=1))
        else:
            vec_m, _ = netEM(torch.cat([img_1, img_2], dim=1))
        pred = netD(vec_c, skip, vec_m)
        recon = netI(vec_c, None, None)
    
    # Reconstruction loss
    l1_loss = F.l1_loss(pred, img_2) + F.l1_loss(recon, img_1)
    # l2_loss = F.mse_loss(pred, img_2)
    
    # Edge loss
    edge_loss = EdgeLoss(pred, img_2, opt.device)
    
    # Perceptual loss
#     perceptual = perceptual_loss(pred, img_2)
    
    # KLD loss
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    # full loss
    loss = l1_loss + 0.1 * edge_loss + opt.kl_weight * kld_loss
#     loss = l1_loss + 0.1 * perceptual + opt.kl_weight * kld_loss
#     loss = l1_loss + opt.kl_weight * kld_loss
#     loss = l2_loss
#     loss = perceptual
    loss.backward()
    optimizerEC.step()
    optimizerEM.step()
    optimizerD.step()
    optimizerI.step()
    
    return l1_loss.item(), edge_loss.item(), kld_loss.item()

### Train identity mapping
def train_identity(img_1, img_2):
    optimizerEC.zero_grad()
    optimizerEM.zero_grad()
    optimizerD.zero_grad()
    
    with torch.set_grad_enabled(True):
        vec_c, skip = netEC(img_1)
        if netEM.vae:
            vec_i_2, _, _, _ = netEM(torch.cat([img_2, img_2], dim=1))
            vec_i_1, _, _, _ = netEM(torch.cat([img_1, img_1], dim=1))
        else:
            vec_i_2, _ = netEM(torch.cat([img_2, img_2], dim=1))
            vec_i_1, _ = netEM(torch.cat([img_1, img_1], dim=1))
        vec_i_1 = vec_i_1.detach()
        pred = netD(vec_c, skip, vec_i_2)
        
    # Reconstruction Loss
    l1_loss = F.l1_loss(pred, img_1)
    
    # Similarity loss between two identity motion vector
    sim_loss = F.l1_loss(vec_i_1, torch.zeros_like(vec_i_1)) + F.l1_loss(vec_i_2, torch.zeros_like(vec_i_2))
    
    # Full loss
    loss = l1_loss + sim_loss
    loss.backward()
    optimizerEC.step()
    optimizerEM.step()
    optimizerD.step()
    
    return l1_loss.item() + sim_loss.item()
    
def test(img_1, img_2):
    with torch.no_grad():
        vec_c, skip = netEC(test_img_1)
        if netEM.vae:
            vec_m, _, _, _ = netEM(torch.cat([test_img_1, test_img_2], dim=1))
        else:
            vec_m, _ = netEM(torch.cat([test_img_1, test_img_2], dim=1))
#         vec_m, _ = netEM(test_img_2)
        pred = netD(vec_c, skip, vec_m)
    return pred



## Training Loop
### Visdom server
vis = visdom.Visdom(env=opt.vis_name)

### Fix test data
test_img_1 = []
test_img_2 = []
for _ in range(4):
    img_1, img_2 = test_dataset[_]
    test_img_1.append(img_1.unsqueeze(0))
    test_img_2.append(img_2.unsqueeze(0))
test_img_1 = torch.cat(test_img_1, dim=0).to(opt.device)
test_img_2 = torch.cat(test_img_2, dim=0).to(opt.device)

### Loss record
loss_record = []

for epoch in tqdm(range(opt.niter), desc='Epoch'):
    netEC.train()
    netEM.train()
    netD.train()
    
    epoch_l1_loss, epoch_edge_loss, epoch_id_loss, epoch_perceptual_loss, epoch_kld_loss = 0, 0, 0, 0, 0
    for i in tqdm(range(opt.epoch_size), desc='Batch'):
        img_1, img_2 = next(train_generator)
        img_1, img_2 = img_1.to(opt.device), img_2.to(opt.device)
        
        l1_loss, edge_loss, kld_loss = train(img_1, img_2)
        epoch_l1_loss += l1_loss
        epoch_edge_loss += edge_loss
        epoch_kld_loss += kld_loss
#         epoch_perceptual_loss += perceptual
        
        
#         if i % 10 == 0:
#             id_loss = train_identity(img_1, img_2)
#             epoch_id_loss += id_loss
        
    netEC.eval()
    netEM.eval()
    netD.eval()
    pred = test(test_img_1, test_img_2)
    vis.images(torch.cat([test_img_1, test_img_2, pred], dim=0))
    
    print('[%02d] l1 loss: %.4f | edge loss: %.4f | kld loss: %.4f | identity loss: %.4f' % (epoch, epoch_l1_loss/opt.epoch_size, epoch_edge_loss/opt.epoch_size, epoch_kld_loss/opt.epoch_size, epoch_id_loss/(opt.epoch_size // 10)))
#     print('[%02d] l1 loss: %.4f | kld loss: %.4f' % (epoch, epoch_l1_loss/opt.epoch_size, epoch_kld_loss/opt.epoch_size))
#     print('[%02d] l2 loss: %.4f' % (epoch, epoch_l2_loss/opt.epoch_size))
#     print('[%02d] l1 loss: %.4f | kld loss: %.4f | identity loss: %.4f' % (epoch, epoch_l1_loss/opt.epoch_size, epoch_kld_loss/opt.epoch_size, epoch_id_loss/(opt.epoch_size // 10)))
#     print('[%02d] perceptual loss: %.4f' % (epoch, epoch_perceptual_loss/opt.epoch_size))

    # Save loss record for plot
    loss_record.append({
        'l1' : epoch_l1_loss/opt.epoch_size,
        'edge' : epoch_edge_loss/opt.epoch_size,
#         'perceptual' : epoch_perceptual_loss/opt.epoch_size,
        'KLD' : epoch_kld_loss/opt.epoch_size,
        'identity' : epoch_id_loss/(opt.epoch_size // 10),
    })

    
    # save the model
    torch.save({
        'netD': netD,
        'netEM': netEM,
        'netEC': netEC,
    }, '%s/model.pth' % opt.log_dir)
    
pkl = open('%s/loss.pkl' % opt.log_dir, 'wb')
pickle.dump(loss_record, pkl)
pkl.close()