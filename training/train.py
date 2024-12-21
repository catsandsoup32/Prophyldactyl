import torch
from torch import nn
from torch.utils.data import DataLoader

from data.dataloader import LichessDataset

device = 'cuda' if torch.cuda.is_available else 'cpu'

train_dataset = LichessDataset(mode='train')
val_dataset = LichessDataset(mode='val')
test_dataset = LichessDataset(mode='test')

# Train loop
def main(num_epochs, model_to_use, lr, experiment_num):
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=8)

