import time
import copy
import torch
import visdom
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
    z = Normal(torch.zeros(batch_size * 32 * 8 * 8), torch.ones(batch_size * 32 * 8 * 8)).sample()
    z = z.view(batch_size, 32, 8, 8).to(DEVICE)
#     c_list = image_encoder(img_1)
    img_pred = decoder(z, img_1)
    z_recon = encoder(img_pred, img_1)
    
    return MSELoss(z_recon, z)

def train_flow(img_1, img_2, model, optimizer_G):
    """
    Work flow:
    inputs(img_2) -> encoder(img_2, img_1) -> z -> decoder(z, img_1) -> flow_recon
    """
    encoder_c, encoder_m, decoder, D = model['encoder_c'], model['encoder_m'], model['decoder'], model['D']
    optimizer_G.zero_grad()
    with torch.set_grad_enabled(True):
        vec_c, skip = encoder_c(img_1)
        if encoder_m.vae:
            vec_m, _, _, _ = encoder_m(torch.cat([img_1, img_2], dim=1))
        else:
            vec_m, _ = encoder_m(torch.cat([img_1, img_2], dim=1))
        img_pred_2 = decoder(vec_c, skip, vec_m)
        
        l1 = l1_loss(img_pred_2, img_2)
        edge = EdgeLoss(img_pred_2, img_2)
        G = -torch.mean(D(torch.cat([img_1, img_pred_2], dim=1)))
        
        loss = l1 + 0.1 * edge + G
        loss.backward()
        optimizer_G.step()
        
    return l1, edge, G

def img_loss(img_1, img_2, img_pred_1, img_pred_2):
    l1 = 1 * l1_loss(img_pred_1, img_1) + 0 * l1_loss(img_pred_2, img_2)
    mse = 1 * MSELoss(img_pred_1, img_1) + 0 * MSELoss(img_pred_2, img_2)
    edge = 1 * EdgeLoss(img_pred_1, img_1) + 0 * EdgeLoss(img_pred_2, img_2)
    return l1, mse, edge

def train_vae(model, img_1, img_2):
    """
    This function is used to train vae based encoder
    """
    encoder_m = model['encoder_m']
    _, mu, log_var, _ = encoder_m(torch.cat([img_1, img_2], dim=1))
    
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return kld

def train_D(img_1, img_2, model, optimizer_D):
    encoder_c, decoder, D = model['encoder_c'], model['decoder'], model['D']
    
    z = Normal(torch.zeros(img_1.shape[0] * 128 * 1 * 1), torch.ones(img_1.shape[0] * 128 * 1 * 1)).sample()
    z = z.view(img_1.shape[0], 128, 1, 1).to(DEVICE)
    optimizer_D.zero_grad()
    with torch.set_grad_enabled(True):
        vec_c, skip = encoder_c(img_1)
        fake_img = decoder(vec_c, skip, z).detach()
        
        # Adversarial loss
        loss_D = -torch.mean(D(torch.cat([img_1, img_2], dim=1))) + torch.mean(D(torch.cat([img_1, fake_img], dim=1)))
        loss_D.backward()
        optimizer_D.step()
    
    # Clip weight in D
    for p in D.parameters():
        p.data.clamp_(-0.01, 0.01)
    return loss_D

def unnorm(inputs):
    out = 0.5 * inputs + 0.5
    out = torch.clamp(out, -1, 1)
    return out
    

def train(model, dataloader, testloader, optimizer_G, optimizer_D, n_epochs=30, batch_size=100, epoch_size=600, env_name='main'):
    since = time.time()
    
    # Visdom server
    vis = visdom.Visdom(env=env_name)
    vis_train = visdom.Visdom(env='train_' + env_name)
    
    # Create fix test batch
    test_img_1 = []
    test_img_2 = []
    for _ in range(4):
        img_1, img_2 = testloader[_]
        test_img_1.append(img_1.unsqueeze(0))
        test_img_2.append(img_2.unsqueeze(0))
    test_img_1 = torch.cat(test_img_1, dim=0).to(DEVICE)
    test_img_2 = torch.cat(test_img_2, dim=0).to(DEVICE)

    best_weight = {
        'encoder_c' : copy.deepcopy(model['encoder_c'].state_dict()),
        'encoder_m' : copy.deepcopy(model['encoder_m'].state_dict()),
        'decoder' : copy.deepcopy(model['decoder'].state_dict())
    }
    min_loss = float('inf')
    loss_record = {
#         'z_loss' : [],
        'l1' : [],
        'edge' : [],
        'D' : [],
        'G' : [],
        'total_loss' : []
    }
    for epoch in tqdm(range(n_epochs), desc='Epoch'):
    
        
#         model['image_encoder'].train()

        running_loss = {
#             'z_loss' : 0.0,
            'l1' : 0.0,
            'edge' : 0.0,
            'D' : 0.0,
            'G' : 0.0,
        }

        # Iterate over dataloader
        for i in tqdm(range(epoch_size), desc='Batch'):
            model['encoder_c'].train()
            model['encoder_m'].train()
            model['decoder'].train()
            model['D'].train()
            img_1, img_2 = next(iter(dataloader))
            
            img_1 = img_1.to(DEVICE)
            img_2 = img_2.to(DEVICE)

            # Forwarding and track history only when training
            loss = {}
            
            # Train D
            loss['D'] = train_D(img_1, img_2, model, optimizer_D)
            loss['l1'], loss['edge'], loss['G'] = train_flow(img_1, img_2, model, optimizer_G)
            
            # Print some sample
            if i % 100 == 0:
                with torch.no_grad():
                    model['encoder_c'].eval()
                    model['encoder_m'].eval()
                    model['decoder'].eval()
                    model['D'].eval()
                    vec_c, skip = model['encoder_c'](img_1)
                    vec_m, _, _, _ = model['encoder_m'](torch.cat([img_1, img_2], dim=1))
                    img_pred = model['decoder'](vec_c, skip, vec_m)
                    vis_train.images(torch.cat([img_1[:4, ...], img_2[:4, ...], img_pred[:4, ...]], dim=0))
                    
            for k in running_loss.keys():
                running_loss[k] += loss[k].item()
        for k in running_loss.keys():
            loss_record[k].append(running_loss[k] / epoch_size)
        epoch_loss = sum(running_loss.values()) / epoch_size
        loss_record['total_loss'].append(epoch_loss)
        print('No.{} total loss: {:.4f}, '.format(epoch, epoch_loss), end='')
#         print('z_loss: {:.4f}, l1: {:.4f}, l2: {:.4f}, edge: {:.4f}, z_cycle: {:.4f}'.format(running_loss['z_loss']/len(dataloader), running_loss['l1']/len(dataloader), running_loss['l2']/len(dataloader), running_loss['edge']/len(dataloader), running_loss['z_cycle']/len(dataloader)))
        print('edge: {:.4f}, l1: {:.4f}, G: {:.4f}, D: {:.4f}'.format(running_loss['edge'] / epoch_size, running_loss['l1'] / epoch_size, running_loss['G'] / epoch_size, running_loss['D'] / epoch_size))
        # deep copy the model
#         if epoch_loss < min_loss:
#             min_loss = epoch_loss
#             best_weight['encoder'] = copy.deepcopy(model['encoder'].state_dict())
#             best_weight['decoder'] = copy.deepcopy(model['decoder'].state_dict())
        with torch.no_grad():
            model['encoder_c'].eval()
            model['encoder_m'].eval()
            model['decoder'].eval()
            model['D'].eval()
            vec_c, skip = model['encoder_c'](test_img_1)
            vec_m, _, _, _ = model['encoder_m'](torch.cat([test_img_1, test_img_2], dim=1))
            img_pred = model['decoder'](vec_c, skip, vec_m)
            vis.images(torch.cat([test_img_1, test_img_2, img_pred], dim=0))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Minimun Loss: {:4f}'.format(min_loss))

    # load best model weights
#     model['encoder'].load_state_dict(best_weight['encoder'])
#     model['decoder'].load_state_dict(best_weight['decoder'])
    return model, loss_record
