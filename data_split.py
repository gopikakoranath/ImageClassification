import torch
import os
# from google.colab import drive
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

def data_split():
    #Defining the transformations
    transform = transforms.Compose([
        transforms.Resize(227),
        transforms.ToTensor()
    ])

    #Downloading the SVHN-train dataset
    train_dataset = SVHN(root='data/', download=True, transform=transform)

    #Downloading the SVHN-test dataset
    Test_dataset = SVHN(root='data/', split='test',download=True, transform=transform)

    #Splitting into train and validation datasets
    torch.manual_seed(64)
    train_ds, val_ds = random_split(train_dataset, [int(0.8*len(train_dataset)), len(train_dataset)-int(0.8*len(train_dataset)), ])
    len(train_ds), len(val_ds)

    # Create a DataLoader for efficient batching
    train_dataloader = torch.utils.data.DataLoader(train_ds,batch_size=512)
    val_dataloader = torch.utils.data.DataLoader(val_ds,batch_size=512)
    test_dataloader = torch.utils.data.DataLoader(Test_dataset,batch_size=512)

    #Check the size of the datasets
    print(len(train_dataloader))
    print(len(val_dataloader))
    print(len(test_dataloader))

    return train_dataloader,val_dataloader,test_dataloader