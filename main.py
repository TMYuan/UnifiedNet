# from model import encoder, decoder, image_encoder
from model import drnet
from util import datasets, plot, kth
from torchvision import transforms
from torch.utils.data import DataLoader
from train import train
import loss
import numpy as np
import torch
import torch.optim as optim
import os

PARAM_SINTEL = {
    'train': '/home/julia0607/MPI-Sintel/new/training',
    'test': '/home/julia0607/MPI-Sintel/new/testing',
    'name': 'MPISintel',
    'dtype': 'final'
}
PARAM_CHAIRS = {
    'train': '/home/anatolios/project/All_in_one/FlyingChairs/',
    'test': '/home/julia0607/Flying Chairs/testing',
    'name': 'FlyingChairs',
    'dtype': 'image'
}

SAVED_PATH = 'saved/1021_7/'
BATCH_SIZE = 100
EPOCH_SIZE = 600
N_EPOCHS = 100
LR = 1e-3
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def prepare_data(param, transformation):
    """
    Return dataloader of given parameters
    """
    # Extract parameters
    path_train, path_test, name, dtype = param['train'], param['test'], param['name'], param['dtype']
    
    # Create dataset for train and test path
    data_train = datasets.Datasets(path_train, name, dtype, transforms=transformation, img_size=224)
#     print(data_train[0][1].shape)
#     data_test = datasets.Datasets(path_test, name, dtype, transforms=transformation)

    # Create dataloader for train and test
    dataloader_train = DataLoader(dataset=data_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
#     dataloader_test = DataLoader(dataset=data_test, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader_train, dataloader_train

if __name__ == '__main__':
    # Prepare dataloader
    trans = transforms.Compose([
                transforms.ToTensor(),
#                 transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
    kth_dataset = kth.KTH(train=True, data_root='./data', seq_len=20, image_size=64, transforms=trans)
    kth_train = DataLoader(dataset=kth_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
    kth_test = kth.KTH(train=False, data_root='./data', seq_len=20, image_size=64, transforms=trans)
    
    if not os.path.exists(SAVED_PATH):
        os.makedirs(SAVED_PATH)

    # Build Model
    model = {
#         'encoder': encoder(channel_in=6, vae=False).to(DEVICE),
#         'decoder': decoder(channel_in=3, f_extracter=True).to(DEVICE),
#         'image_encoder': image_encoder().to(DEVICE)
        'encoder_c': drnet.Encoder(1, 128).to(DEVICE),
        'encoder_m': drnet.Encoder(2, 128, vae=True).to(DEVICE),
        'decoder': drnet.Decoder(128+128, 1).to(DEVICE),
        'D': drnet.Discriminator(2, (1, 64, 64)).to(DEVICE)
    }
#     params = []
#     for m in model.values():
#         params += list(m.parameters())
    params = list(model['encoder_c'].parameters()) + list(model['encoder_m'].parameters()) + list(model['decoder'].parameters())
    optimizer_G = optim.RMSprop(params, lr=LR)
    optimizer_D = optim.RMSprop(model['D'].parameters(), lr=LR)
    
    # Training Procedure
    model, loss_record = train(model, kth_train, kth_test, optimizer_G, optimizer_D, N_EPOCHS, batch_size=BATCH_SIZE, epoch_size=EPOCH_SIZE, env_name=SAVED_PATH)
    torch.save(model['encoder_c'].state_dict(), os.path.join(SAVED_PATH, 'weight_encoder_c.pt'))
    torch.save(model['encoder_m'].state_dict(), os.path.join(SAVED_PATH, 'weight_encoder_m.pt'))
    torch.save(model['decoder'].state_dict(), os.path.join(SAVED_PATH, 'weight_decoder.pt'))
    torch.save(model['D'].state_dict(), os.path.join(SAVED_PATH, 'weight_D.pt'))
#     torch.save(model['image_encoder'].state_dict(), os.path.join(SAVED_PATH, 'weight_image_encoder.pt'))
    plot.draw(loss_record, SAVED_PATH)