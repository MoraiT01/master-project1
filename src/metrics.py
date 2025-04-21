"""
    All used metrics are defined here.
"""

# import required libraries
import numpy as np

from typing import Literal, Tuple, Dict
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# my
from training import model_params
from mlp_dataclass import MNIST_CostumDataset
from my_random import shared_random_state
import math
import datetime
from tqdm import tqdm

torch.manual_seed(100)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    _, l = torch.max(labels, dim=1)
    correct = torch.sum(preds == l).item()
    return torch.tensor(correct / len(preds))

def validation_step(model, batch):
    images, labels = batch
    images, labels = images.to(DEVICE), labels.to(DEVICE)
    out = model(images)                    
    loss = F.cross_entropy(out, labels)   
    acc = accuracy(out, labels)
    return {'Loss': loss.detach(), 'Acc': acc}

def validation_epoch_end(model, outputs):
    batch_losses = [x['Loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()   
    batch_accs = [x['Acc'] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()      
    return {'Loss': epoch_loss.item(), 'Acc': epoch_acc.item()}

@torch.no_grad()
def evaluate_loss_accs(model, val_loader):
    model.eval()
    outputs = [validation_step(model, batch) for batch in val_loader]
    return validation_epoch_end(model, outputs)