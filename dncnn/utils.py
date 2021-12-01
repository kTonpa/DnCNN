"""
Project: dncnn
Author: khalil MEFTAH
Date: 2021-11-26
DnCNN: Deep Neural Convolutional Network for Image Denoising model utility functions
"""

# Imports
import torch

class AWGN(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
