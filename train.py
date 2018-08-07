import time
import copy
import torch
from tqdm import tqdm
from torch.distributions.normal import Normal
from loss import MSELoss, SmoothL1Loss, EdgeLoss, L1Loss

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_z(model, images, batch_size, tmp):
    """
    Work flow:
    z -> decoder(z, lbl) -> flow -> encoder(flow, lbl) -> z_recon
    """
    encoder, decoder = model['encoder'], model['decoder']
    # Get condition from images
    _, c_2, c_4, c_8 = encoder(tmp, images)
    c_2, c_4, c_8 = c_2.detach(), c_4.detach(), c_8.detach()
    
    # random noise
    z = Normal(torch.zeros(batch_size * 14 * 14), torch.ones(batch_size * 14 * 14)).sample()
    z = z.view(batch_size, 1, 14, 14).to(DEVICE)
    flows = decoder(z, images, c_2, c_4, c_8)
    z_recon, _, _, _ = encoder(flows, images)
    return MSELoss(z_recon, z)

def train_flow(model, flows, images):
    """
    Work flow:
    inputs(flow) -> encoder(input, lbl) -> z -> decoder(z, lbl) -> flow_recon
    """
    encoder, decoder = model['encoder'], model['decoder']
    z, c_2, c_4, c_8 = encoder(flows, images)
    flows_recon = decoder(z, images, c_2, c_4, c_8)
    return MSELoss(flows_recon, flows)

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
        for i, (img, flow) in enumerate(tqdm(dataloader, desc='Batch')):
            assert img.shape[1] == 3
            assert flow.shape[1] == 2
            
#             if i >= 1000:
#                 break
            img = img.to(DEVICE)
            flow = flow.to(DEVICE)
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
                loss['z_loss'] = train_z(model, img, batch_size, flow)
                loss['recon_loss'] = train_flow(model, flow, img)
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
        # deep copy the model
        if epoch_loss < min_loss:
            min_loss = epoch_loss
            best_weight['encoder'] = copy.deepcopy(model['encoder'].state_dict())
            best_weight['decoder'] = copy.deepcopy(model['decoder'].state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Minimun Loss: {:4f}'.format(min_loss))

    # load best model weights
    model['encoder'].load_state_dict(best_weight['encoder'])
    model['decoder'].load_state_dict(best_weight['decoder'])
    return model, loss_record
