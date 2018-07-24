import time
import copy
import torch
from tqdm import tqdm

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(model, dataloader, criterion, optimizer, scheduler, n_epochs=30):
    since = time.time()

    best_weight = copy.deepcopy(model.state_dict())
    min_loss = float('inf')

    for epoch in tqdm(range(n_epochs), desc='Epoch'):
        for phase in tqdm(['train', 'val'], desc='Phase'):
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()
            running_loss = 0.0

            # Iterate ove dataloader
            for i, (inputs, labels) in enumerate(tqdm(dataloader[phase], desc='Batch')):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                # zero the parameter gradients
                optimizer.zero_grad()

                # Forwarding and track history only when training
                with torch.set_grad_enabled(phase == 'train'):
                    # TODO: build the flow of input and output on the model.
                    # output = model(inputs)
                    # loss = criterion(outputs, labels) ...

                    # backward & optimize if phase == train
                    if phase == 'train':
                        # TODO: update loss
                        # loss.backward
                        # optimizer.step() ...
                
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloader[phase])
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss < min_loss:
                min_loss = epoch_loss
                best_weight = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Minimun Loss: {:4f}'.format(min_loss))

    # load best model weights
    model.load_state_dict(best_weight)
    return model
