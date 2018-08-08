from model import encoder, decoder, image_encoder, discriminator
from util import datasets, plot
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
SAVED_PATH = './saved/0806/'

BATCH_SIZE = 20
N_EPOCHS = 5
LR = 1e-2
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
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    fc_train, _ = prepare_data(PARAM_CHAIRS, trans)

    # Build Model
    model = {
        'encoder': encoder(vae=False).to(DEVICE),
        'decoder': decoder().to(DEVICE),
        'image_encoder': image_encoder().to(DEVICE),
        'discriminator' : discriminator().to(DEVICE)
    }
    params = []
    for k in ['encoder', 'decoder', 'image_encoder']:
        params += list(model[k].parameters())
#     optimizer = optim.SGD(params, lr=LR, momentum=0.9, weight_decay=0.1)
    optimizer = {
        'G' : optim.Adam(params),
        'D' : optim.Adam(model['discriminator'].parameters(), lr=1e-5)
    }
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer['G'], patience=3)
    
    # Training Procedure
    model, loss_record = train(model, fc_train, optimizer, scheduler, N_EPOCHS, batch_size=BATCH_SIZE)
    if not os.path.exists(SAVED_PATH):
        os.makedirs(SAVED_PATH)
    torch.save(model['encoder'].state_dict(), os.path.join(SAVED_PATH, 'weight_encoder.pt'))
    torch.save(model['decoder'].state_dict(), os.path.join(SAVED_PATH, 'weight_decoder.pt'))
    torch.save(model['image_encoder'].state_dict(), os.path.join(SAVED_PATH, 'weight_image_encoder.pt'))
    torch.save(model['discriminator'].state_dict(), os.path.join(SAVED_PATH, 'weight_discriminator.pt'))
    plot.draw(loss_record, SAVED_PATH)