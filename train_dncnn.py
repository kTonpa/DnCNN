"""
Project: dncnn
Author: khalil MEFTAH
Date: 2021-11-26
DnCNN: Deep Neural Convolutional Network for Image Denoising model training implementation
"""

# Imports

import argparse
import wandb
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms as T

from dncnn.dncnn import DnCNN
from dncnn.loader import NoisyDataset
from dncnn.utils import AWGN

# arguments parsing

parser = argparse.ArgumentParser(description='DnCNN Training')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs for training')
parser.add_argument('--batch-size', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--output_file_name', type=str, default = "DnCNN", help='output_file_name')
parser.add_argument('--wandb-name', type=str, default='dncnn', help='name of wandb run')

args = parser.parse_args()

# helper functions

def create_img_transform(image_width, std_value):
    transform = T.Compose([
                    T.CenterCrop((image_width, image_width)),
                    T.ToTensor(),
                    AddGaussianNoise(0., std_value)
            ])
    return transform


# constants

EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LR = args.lr
OUTPUT_FILE_NAME = args.output_file_name
WANDB_NAME = args.wandb_name

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# create dataset and dataloader

ds = NoisyDataset(root_dir='data/train/', transform=None, shuffle=True)

assert len(ds) > 0, 'dataset is empty'
print(f'{len(ds)} images found for training.')

dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

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

# trainig

for epoch in range(EPOCHS):
    for i, (images, _) in enumerate(dl):
        if i % 10 == 0:
            t = time.time()

        images = images.to(device)
        loss = dncnn(images, return_loss=True)
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

        if i % 10 == 9:
            sample_per_sec = BATCH_SIZE * 10 / (time.time() - t)
            log["sample_per_sec"] = sample_per_sec
            print(epoch, i, f'sample_per_sec - {sample_per_sec}')

        wandb.log(log)

    # save trained model to wandb as an artifact every epoch's end
    
    model_artifact = wandb.Artifact('trained-dncnn', type='model', metadata=dict(model_config))
    model_artifact.add_file(f'{OUTPUT_FILE_NAME}.pt')
    run.log_artifact(model_artifact)


save_model(f'./{OUTPUT_FILE_NAME}-final.pt', epoch=epoch)
wandb.save(f'./{OUTPUT_FILE_NAME}.pt')
model_artifact = wandb.Artifact('trained-dncnn', type='model', metadata=dict(model_config))
model_artifact.add_file(f'{OUTPUT_FILE_NAME}-final.pt')
run.log_artifact(model_artifact)

wandb.finish()
