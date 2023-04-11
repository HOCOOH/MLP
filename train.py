import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from mlp import MLP
from get_data import get_data

def train_one_epoch(model, optimizer, loss_func, epoch, n_batch, train_loader, n_epoch):
    train_loss = 0
    total_loss = 0

    with tqdm(total=n_batch, desc=f'Epoch{epoch + 1}/{n_epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, (data, target) in enumerate(train_loader):
            output = model(data)

            loss = loss_func(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            train_loss = total_loss / (iteration + 1)

            pbar.set_postfix(**{"train_loss": train_loss})
            pbar.update(1)  # 更新进度条

    return train_loss


