import time
import copy
import torch
from tqdm import tqdm
from torch.distributions.normal import Normal
from loss import MSELoss, SmoothL1Loss, EdgeLoss

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_z(model, images, batch_size):
    """
    Work flow:
    z -> decoder(z, lbl) -> flow -> encoder(flow, lbl) -> z_recon
    """
    encoder, decoder = model['encoder'], model['decoder']
    z = Normal(torch.zeros(batch_size * 14 * 14), torch.ones(batch_size * 14 * 14)).sample()
    z = z.view(batch_size, 1, 14, 14).to(DEVICE)
    flows = decoder(z, images)
    z_recon = encoder(flows, images)
    return MSELoss(z_recon, z)

def train_flow(model, flows, images):
    """
    Work flow:
    inputs(flow) -> encoder(input, lbl) -> z -> decoder(z, lbl) -> flow_recon
    """
    encoder, decoder = model['encoder'], model['decoder']
    z = encoder(flows, images)
    flows_recon = decoder(z, images)
    return SmoothL1Loss(flows_recon, flows) + EdgeLoss(flows_recon, flows)

def train(model, dataloader, optimizer, scheduler, n_epochs=30, batch_size=20):
    since = time.time()

    best_weight = {
        'encoder' : copy.deepcopy(model['encoder'].state_dict()),
        'decoder' : copy.deepcopy(model['decoder'].state_dict())
    }
    min_loss = float('inf')

    for epoch in tqdm(range(n_epochs), desc='Epoch'):
    
        model['encoder'].train()
        model['decoder'].train()

        running_loss = {
            'z_loss' : 0.0,
            'recon_loss' : 0.0 
        }

        # Iterate over dataloader
        for i, (img, flow) in enumerate(tqdm(dataloader, desc='Batch')):
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
                loss['z_loss'] = train_z(model, img, batch_size)
                loss['recon_loss'] = train_flow(model, flow, img)
#                 print('z_loss: {}'.format(loss['z_loss'].item()))
#                 print('recon_loss: {}'.format(loss['recon_loss'].item()))
                # backward & optimize if phase == train
                total_loss = loss['z_loss'] + loss['recon_loss']
                total_loss.backward()
                optimizer.step()
                scheduler.step(total_loss)
            
            for k in running_loss.keys():
                running_loss[k] += loss[k].item()

        epoch_loss = sum(running_loss.values()) / len(dataloader)
        print('No.{} Loss: {:.4f}'.format(epoch, epoch_loss))
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
    model['decoder'].load_state_dict(best_weight['encoder'])
    return model
