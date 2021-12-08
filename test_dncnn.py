"""
Project: dncnn
Author: khalil MEFTAH
Date: 2021-11-26
DnCNN: Deep Neural Convolutional Network for Image Denoising model testing implementation
"""

# Imports

import argparse
import time
import numpy as np

from pathlib import Path

import torch
from PIL import Image

from dncnn.dncnn import DnCNN

# arguments parsing

parser = argparse.ArgumentParser(description='DnCNN Testing')
parser.add_argument('--image_path', type=str, required=True, help='test image path')
parser.add_argument('--model_path', type=str, required=True, help='pretrained model path')
parser.add_argument('--output_folder', type=str, required=True, help='output folder')

args = parser.parse_args()

# helper fns

def exists(val):
    return val is not None

def load_image(path):
    img = Image.open(path).convert('L')
    img = np.array(img)
    img = torch.from_numpy(img)
    img = torch.unsqueeze(img, dim=0)
    img = img.float()
    img = torch.unsqueeze(img, dim=0)
    img = img / 255.
    return img

def save_image(path, img):
    img = img.detach().cpu().numpy()
    img = np.squeeze(img)
    img = np.clip(img, 0, 1)
    img = img * 255.
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    img.save(path+'dncnn_output.jpg')

# constants

IMAGE_PATH = args.image_path
MODEL_PATH = args.model_path
OUTPUT_FOLDER = args.output_folder

# load DnCNN model

dncnn = None
if exists(MODEL_PATH):
    dncnn_path = Path(MODEL_PATH)
    assert dncnn_path.exists(), 'pretrained dncnn must exist'

    dncnn = torch.load(str(dncnn_path))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

noisy_img = load_image(IMAGE_PATH)
noisy_img = noisy_img.to(device)

dncnn_clean_img = dncnn.denoise(noisy_img)

save_image(OUTPUT_FOLDER, dncnn_clean_img)
