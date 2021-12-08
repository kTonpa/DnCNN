"""
Project: dncnn
Author: khalil MEFTAH
Date: 2021-11-26
DnCNN: Deep Neural Convolutional Network for Image Denoising model training implementation
"""

# Imports

import argparse
import time
import wandb
import numpy as np

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from PIL import Image
from torchvision.utils import make_grid
from torchvision import transforms as T

from dncnn.dncnn import DnCNN
from dncnn.loader import NoisyDataset
from dncnn.utils import AWGN

# arguments parsing

parser = argparse.ArgumentParser(description='DnCNN Training')
parser.add_argument('--image_folder', type=str, required=True,
                    help='path to your folder of images for learning the DnCNN')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs for training')
parser.add_argument('--batch-size', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--output_file_name', type=str, default = "DnCNN", help='output_file_name')
parser.add_argument('--wandb-name', type=str, default='dncnn', help='name of wandb run')

args = parser.parse_args()

# helper functions

def create_img_transform(image_width, crop_size):
    transform = T.Compose([
                    T.CenterCrop((image_width, image_width)),
                    T.RandomCrop((crop_size, crop_size)),
                    # T.ToTensor()
                    # AWGN(0., std_value)
            ])
    return transform

def img_transform():
    transform = T.Compose([
                    T.ToTensor()
            ])
    return transform

# create python function to load image in grayscale and transform it to batch tensor 

def load_image(path, std, mean):
    img = Image.open(path).convert('L')
    img = np.array(img)
    img = torch.from_numpy(img)
    img = torch.unsqueeze(img, dim=0)
    img = img.float()
    img_y = img + std * torch.randn(img.shape) + mean
    img = torch.unsqueeze(img, dim=0)
    img_y = torch.unsqueeze(img_y, dim=0)
    img = img / 255.
    img_y = img_y / 255.
    return img, img_y

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

clean_img, noisy_img = load_image("/content/BSR/BSDS500/data/images/train/100075.jpg", 10., 0.)
clean_img = clean_img.to(device)
noisy_img = noisy_img.to(device)

# constants

ROOT_DIR = args.image_folder
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LR = args.lr
OUTPUT_FILE_NAME = args.output_file_name
WANDB_NAME = args.wandb_name

# create dataset and dataloader

ds = NoisyDataset(root_dir=ROOT_DIR, std=1.0, mean=0.0, transform=create_img_transform(180, 40), shuffle=True)

ds_test = NoisyDataset(root_dir="/content/BSR/BSDS500/data/images/test", std=10.0, mean=0.0, transform=img_transform(), shuffle=True)

assert len(ds) > 0, 'dataset is empty'
print(f'{len(ds)} images found for training.')

dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

dl_test = DataLoader(ds_test, batch_size=1, shuffle=True, drop_last=False)


dncnn = DnCNN(
    num_layers=17,
    num_features=64,
    kernel_size=3,
    padding=1,
    image_channels=1,
    image_size=64
)

dncnn = dncnn.to(device)

# optimizer

opt = Adam(dncnn.parameters(), lr=LR)

# experiment tracker

model_config = dict(
    batch_size = BATCH_SIZE,
    learning_rate = LR,
    epochs = EPOCHS
)

run = wandb.init(
    project=args.wandb_name,  # 'dncnn' by default
    config=model_config
)

def save_model(path):
    # save_obj = dncnn.state_dict()
    torch.save(dncnn, path)

# save_model(f'./{OUTPUT_FILE_NAME}.pt')


# trainig

for epoch in range(EPOCHS):
    for i, (clean, noisy) in enumerate(dl):
        if i % 10 == 0:
            t = time.time()

        clean, noisy = map(lambda t: t.to(device), (clean, noisy))
        loss = dncnn(noisy, return_loss=True, x=clean)
        loss.backward()
        opt.step()
        opt.zero_grad()

        log = {}

        if i % 10 == 0:
            log = {
                'loss': loss.item(),
                'epoch': epoch,
                'iter': i
            }

        if i % 100 == 0:
            # denoise the image

            clean_img_dncnn = dncnn.denoise(noisy_img)

            log = {
                **log,
            }

            compare_image = torch.cat((noisy_img, clean_img_dncnn, clean_img), dim=0)

            # save the image to wandb
             
            grid = make_grid(compare_image, value_range=(0.0, 1.0), normalize=True)
            ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            pil_image = Image.fromarray(ndarr)

            log['image'] = wandb.Image(pil_image, caption="Denoised image")

        if i % 10 == 9:
            sample_per_sec = BATCH_SIZE * 10 / (time.time() - t)
            log["sample_per_sec"] = sample_per_sec
            print(epoch, i, f'sample_per_sec - {sample_per_sec}')

        wandb.log(log)

    # save trained model to wandb as an artifact every epoch's end
    
    model_artifact = wandb.Artifact('trained-dncnn', type='model', metadata=dict(model_config))
    run.log_artifact(model_artifact)


save_model(f'./{OUTPUT_FILE_NAME}-final.pt')
model_artifact = wandb.Artifact('trained-dncnn', type='model', metadata=dict(model_config))
run.log_artifact(model_artifact)

wandb.finish()
