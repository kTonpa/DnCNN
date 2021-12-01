"""
Project: dncnn
Author: khalil MEFTAH
Date: 2021-11-26
DnCNN: Deep Neural Convolutional Network for Image Denoising model implementation
"""

import torch
from torch import nn
import torch.nn.functional as F

class DnCNN(nn.Module):
    def __init__(
        self,
        num_layers=17,
        num_features=64,
        kernel_size=3,
        padding=1,
        image_channels=1,
        image_size=64
    ):
        super(DnCNN, self).__init__()
        layers = []
        
        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=num_features, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=kernel_size, padding=padding, bias=True))
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.Conv2d(in_channels=num_features, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=True))
        
        self.dncnn = nn.Sequential(*layers)

    def forward(self, y, x, return_loss = False):
        n = self.dncnn(y)

        if return_loss:
            return n - (y-x)
        
        return y-n
