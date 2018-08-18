import time
import copy
import torch
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
from torch.distributions.normal import Normal
from loss import MSELoss, SmoothL1Loss, EdgeLoss, L1Loss

# SAVED_PATH = './saved/0811/'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_z(model, images, batch_size):
    """
    Work flow:
    conditions(image) -> image_encoder -> features
    z -> decoder(z, features) -> flow -> encoder(flow, image) -> z_recon
    """
    encoder, decoder, image_encoder = model['encoder'], model['decoder'], model['image_encoder']
    # Get condition from images
    c_2, c_4, c_8, c_16 = image_encoder(images)
    c_2, c_4, c_8, c_16 = c_2.detach(), c_4.detach(), c_8.detach(), c_16.detach()
    
    # random noise
    z = Normal(torch.zeros(batch_size * 14 * 14), torch.ones(batch_size * 14 * 14)).sample()
    z = z.view(batch_size, 1, 14, 14).to(DEVICE)
    flows = decoder(z, images, c_2, c_4, c_8, c_16)
    z_recon = encoder(flows, images)
    return MSELoss(z_recon, z)

def train_flow(model, flows, images, images_2):
    """
    Work flow:
    inputs(flow) -> encoder(input, lbl) -> z
    conditions(images) -> image_encoder -> features
    decoder(z, features) -> flow_recon
    """
    encoder, decoder, image_encoder = model['encoder'], model['decoder'], model['image_encoder']
    z = encoder(flows, images)
    c_2, c_4, c_8, c_16 = image_encoder(images)
    flows_recon = decoder(z, images, c_2, c_4, c_8, c_16)
    return warp_loss(images, images_2, flows_recon)

def refinement(r_2, r_4, r_8, r, flows):
    f_2 = F.avg_pool2d(flows, 2)
    f_4 = F.avg_pool2d(flows, 4)
    f_8 = F.avg_pool2d(flows, 8)
    return MSELoss(r, flows) + MSELoss(r_2, f_2) + MSELoss(r_4, f_4) + MSELoss(r_8, f_8)

def warp_loss(img, img_2, flow):
    # renormalize
    flow = flow * 20.0
    flow = flow / 224
    
    grid_x, grid_y = np.meshgrid(np.linspace(-1, 1, 224), np.linspace(-1, 1, 224))  # (h, w)
    grid_x = torch.from_numpy(grid_x.astype('float32')).to(DEVICE)
    grid_y = torch.from_numpy(grid_y.astype('float32')).to(DEVICE)
    
    grid = torch.stack((grid_x, grid_y), 2).unsqueeze(0).repeat(flow.shape[0], 1, 1, 1)
    grid[..., 0] += flow[:, 0, ...]
    grid[..., 1] += flow[:, 1, ...]
    
    img_warp = F.grid_sample(img_2, grid, padding_mode='border')
    return MSELoss(img_warp, img)
    

def train(model, dataloader, optimizer, scheduler, n_epochs=30, batch_size=20, saved_path='./saved'):
    since = time.time()

    best_weight = {
        'encoder' : copy.deepcopy(model['encoder'].state_dict()),
        'decoder' : copy.deepcopy(model['decoder'].state_dict()),
        'image_encoder': copy.deepcopy(model['image_encoder'].state_dict())
    }
    min_loss = float('inf')
    loss_record = {
        'z_loss' : [],
        'recon_loss' : [],
        'total_loss' : []
    }
    for epoch in tqdm(range(n_epochs), desc='Epoch'):
    
        model['encoder'].train()
        model['decoder'].train()
        model['image_encoder'].train()

        running_loss = {
            'z_loss' : 0.0,
            'recon_loss' : 0.0 
        }

        # Iterate over dataloader
        for i, (img, flow, img_2) in enumerate(tqdm(dataloader, desc='Batch')):
            assert img.shape[1] == 3
            assert flow.shape[1] == 2
            
#             if i >= 5:
#                 break
            img = img.to(DEVICE)
            flow = flow.to(DEVICE)
            img_2 = img_2.to(DEVICE)
#             print('img shape: {}'.format(img.shape))
#             print('flow shape: {}'.format(flow.shape))
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # Forwarding and track history only when training
            loss = {}
            with torch.set_grad_enabled(True):
                # TODO: build the flow of input and output on the model.
                # output = model(inputs)
                # loss = criterion(outputs, labels) ...
                loss['z_loss'] = train_z(model, img, batch_size)
                loss['recon_loss'] = train_flow(model, flow, img, img_2)
#                 print('z_loss: {}'.format(loss['z_loss'].item()))
#                 print('recon_loss: {}'.format(loss['recon_loss'].item()))
                # backward & optimize if phase == train
                total_loss = 0 * loss['z_loss'] + loss['recon_loss']
                total_loss.backward()
                optimizer.step()
#                 scheduler.step(total_loss)
            
            for k in running_loss.keys():
                running_loss[k] += loss[k].item()
        for k in running_loss.keys():
            loss_record[k].append(running_loss[k] / len(dataloader))
        epoch_loss = sum(running_loss.values()) / len(dataloader)
        loss_record['total_loss'].append(epoch_loss)
        print('No.{} total loss: {:.4f}, '.format(epoch, epoch_loss), end='')
        print('z_loss: {:.4f}, recon_loss: {:4f}'.format(running_loss['z_loss']/len(dataloader), running_loss['recon_loss']/len(dataloader)))
        
        torch.save(model['encoder'].state_dict(), os.path.join(saved_path, 'weight_encoder.pt'))
        torch.save(model['decoder'].state_dict(), os.path.join(saved_path, 'weight_decoder.pt'))
        torch.save(model['image_encoder'].state_dict(), os.path.join(saved_path, 'weight_image_encoder.pt'))
        # deep copy the model
#         if epoch_loss < min_loss:
#             min_loss = epoch_loss
#             best_weight['encoder'] = copy.deepcopy(model['encoder'].state_dict())
#             best_weight['decoder'] = copy.deepcopy(model['decoder'].state_dict())
#             best_weight['image_encoder'] = copy.deepcopy(model['image_encoder'].state_dict())

#     time_elapsed = time.time() - since
#     print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
#     print('Minimun Loss: {:4f}'.format(min_loss))

    # load best model weights
#     model['encoder'].load_state_dict(best_weight['encoder'])
#     model['decoder'].load_state_dict(best_weight['decoder'])
#     model['image_encoder'].load_state_dict(best_weight['image_encoder'])
    return model, loss_record
