"""
Project: dncnn
Author: khalil MEFTAH
Date: 2021-11-26
DnCNN: Deep Neural Convolutional Network for Image Denoising data loader implementation
"""

# Imports

import numpy as np
from PIL import Image, UnidentifiedImageError

from pathlib import Path
from random import randint

import torch
from torch.utils.data import Dataset

class NoisyDataset(Dataset):
    def __init__(self, root_dir, transform=None, shuffle=False):
        super(NoisyDataset, self).__init__()
        root = Path(root_dir)
        self.transform = transform
        self.shuffle = shuffle

        self.noisy_images = list(*root.glob('*.png'))

    def __len__(self):
        return len(self.noisy_images)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, idx):
        img_path = self.noisy_images[idx]

        try:
            img = Image.open(img_path).convert('L')
            img = np.array(img)
            img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img)

            if self.transform:
                img = self.transform(img)

        except UnidentifiedImageError as corrupt_image_exceptions:
            print(f"An exception occurred trying to load file {img_path}.")
            print(f"Skipping index {idx}.")
            return self.skip_sample(idx)

        # success
        
        return img
