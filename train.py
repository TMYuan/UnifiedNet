import time
import copy
import torch
from tqdm import tqdm
from torch.distributions.normal import Normal
from loss import MSELoss, SmoothL1Loss, EdgeLoss, L1Loss
from torch.nn.functional import binary_cross_entropy, l1_loss

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_z(model, img_1, batch_size):
    """
    Work flow:
    z -> decoder(z, img_1) -> img_pred -> encoder(img_pred, img_1) -> z_recon
    """
    encoder, decoder = model['encoder'], model['decoder']
    z = Normal(torch.zeros(batch_size * 1 * 14 * 14), torch.ones(batch_size * 1 * 14 * 14)).sample()
    z = z.view(batch_size, 1, 14, 14).to(DEVICE)
    img_pred = decoder(z, img_1)
    z_recon = encoder(img_pred, img_1)
    return MSELoss(z_recon, z)

def train_flow(model, img_1, img_2):
    """
    Work flow:
    inputs(img_2) -> encoder(img_2, img_1) -> z -> decoder(z, img_1) -> flow_recon
    """
    encoder, decoder = model['encoder'], model['decoder']
    z = encoder(img_2, img_1)
    img_pred = decoder(z, img_1)
    return 0.9 * l1_loss(img_pred, img_2) + 0.1 * EdgeLoss(img_pred, img_2)

def train(model, dataloader, optimizer, scheduler, n_epochs=30, batch_size=20):
    since = time.time()

    best_weight = {
        'encoder' : copy.deepcopy(model['encoder'].state_dict()),
        'decoder' : copy.deepcopy(model['decoder'].state_dict())
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

        running_loss = {
            'z_loss' : 0.0,
            'recon_loss' : 0.0 
        }

        # Iterate over dataloader
        for i, (img_1, img_2) in enumerate(tqdm(dataloader, desc='Batch')):
#             assert img_1.shape[1] == 3 and flow.shape[1] == 2
            
            
#             if i >= 10:
#                 break
            img_1 = img_1.to(DEVICE)
            img_2 = img_2.to(DEVICE)

            # zero the parameter gradients
            optimizer.zero_grad()
            
            # Forwarding and track history only when training
            loss = {}
            with torch.set_grad_enabled(True):
                
                loss['z_loss'] = train_z(model, img_1, batch_size)
                loss['recon_loss'] = train_flow(model, img_1, img_2)

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
        # deep copy the model
#         if epoch_loss < min_loss:
#             min_loss = epoch_loss
#             best_weight['encoder'] = copy.deepcopy(model['encoder'].state_dict())
#             best_weight['decoder'] = copy.deepcopy(model['decoder'].state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Minimun Loss: {:4f}'.format(min_loss))

    # load best model weights
#     model['encoder'].load_state_dict(best_weight['encoder'])
#     model['decoder'].load_state_dict(best_weight['decoder'])
    return model, loss_record
