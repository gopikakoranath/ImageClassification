import torch
import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import random_split
from torchvision.datasets import SVHN
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

def train_loop(tr_dataloader, vl_dataloader, model, loss_fn, optimizer):
    tr_size = len(tr_dataloader.dataset)
    vl_size = len(vl_dataloader.dataset)
    tr_epoch_loss = 0.0
    vl_epoch_loss = 0.0
    vl_correct = 0
    
    for X, y in tr_dataloader:
        # print(X.size(0))
        # Compute prediction and loss
        X=X.to(device="cuda")
        pred = model(X)
        y = y.to(device="cuda")
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tr_epoch_loss += loss.item()*X.size(0)
    
    for X, y in vl_dataloader:
        X=X.to(device='cuda')
        result= model(X)
        y = y.to(device="cuda")
        loss = loss_fn(result, y)
        
        vl_epoch_loss += loss.item()*X.size(0)
        vl_correct += (result.argmax(1) == y).type(torch.float).sum().item()
    
    vl_correct/=vl_size
    print(f'Train epoch loss is {tr_epoch_loss/tr_size}')
    print(f'Validation epoch loss is {vl_epoch_loss/vl_size}')
    print(f"Validation Accuracy: {(100*vl_correct):>0.1f}%, Avg loss: {vl_epoch_loss/vl_size:>8f} \n")
    return tr_epoch_loss,vl_epoch_loss,vl_correct
