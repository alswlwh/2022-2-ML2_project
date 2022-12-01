import os
import torch


def load_checkpoint(checkpoint_dir, title, model, optimizer):
    checkpoint_path = f'{checkpoint_dir}/{title}.pth'
    state_dict = torch.load(checkpoint_path)
    start_epoch = state_dict['epoch']
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    return start_epoch


def save_checkpoint(checkpoint_dir, title, model, optimizer, epoch):
    os.makedirs(checkpoint_dir, exist_ok=True)
    state_dict = {
        'epoch': epoch + 1,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    checkpoint_path = f'{checkpoint_dir}/{title}.pth'
    torch.save(state_dict, checkpoint_path)

    
def save_pretrained(checkpoint_dir, title, model):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = f'{checkpoint_dir}/{title}.pth'
    torch.save(model.state_dict(), checkpoint_path)