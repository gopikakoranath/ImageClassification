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

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        
        # Define the layers of the network
        self.AlexNet=nn.Sequential(
        nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        # nn.LocalResponseNorm(size=96),
        nn.BatchNorm2d(96),

        nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        # nn.LocalResponseNorm(size=256),
        nn.BatchNorm2d(256),
        
        nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),

        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Flatten(),

        
        nn.Linear(in_features=256*6*6, out_features=4096),
        nn.ReLU(),
        nn.Linear(in_features=4096, out_features=4096),
        nn.ReLU(),
        nn.Linear(in_features=4096, out_features=10),
        nn.ReLU())
        
    def forward(self, x):
        # Define the forward pass of the network
        torch.manual_seed(64)
        x=self.AlexNet(x)

        return x
