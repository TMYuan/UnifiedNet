from model import encoder, decoder
from util import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import loss
import train
import numpy as np
import torch
import torch.optim as optim

PARAM_SINTEL = {
    'train': '/home/julia0607/MPI-Sintel/new/training',
    'test': '/home/julia0607/MPI-Sintel/new/testing',
    'name': 'MPISintel',
    'dtype': 'final'
}
PARAM_CHAIRS = {
    'train': '/home/julia0607/Flying Chairs/training',
    'test': '/home/julia0607/Flying Chairs/testing',
    'name': 'FlyingChairs',
    'dtype': 'image'
}

BATCH_SIZE = 20
N_EPOCHS = 30
LR = 1e-1
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def prepare_data(param, transformation):
    """
    Return dataloader of given parameters
    """
    # Extract parameters
    path_train, path_test, name, dtype = param['train'], param['test'], param['name'], param['dtype']
    
    # Create dataset for train and test path
    data_train = datasets.Datasets(path_train, name, dtype, transforms=transformation)
    data_test = datasets.Datasets(path_test, name, dtype, transforms=transformation)

    # Create dataloader for train and test
    dataloader_train = DataLoader(dataset=data_train, batch_size=BATCH_SIZE, suffle=True)
    dataloader_test = DataLoader(dataset=data_test, batch_size=BATCH_SIZE, suffle=True)
    return dataloader_train, dataloader_test

if __name__ == '__main__':
    # Prepare dataloader
    transformation = transforms.Compose([transforms.ToTensor()])
    fc_train, fc_test = prepare_data(PARAM_CHAIRS, transformation)

    # Build Model
    model = {
        'encoder': encoder(vae=False).to(DEVICE),
        'decoder': decoder().to(DEVICE)
    }
    params = []
    for m in model.items():
        params += list(m.parameters())
    optimizer = optim.SGD(params, lr=LR, momentum=0.9, weight_decay=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    
    # Training Processing