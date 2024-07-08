#!/usr/bin/env python

import argparse
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import transforms
import math
from tqdm import tqdm
import os
import sys
sys.path.append(".")
import utils
import models.builder as builder
import dataloader

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_args():
    print('Parsing argumments')
    parser = argparse.ArgumentParser(description='Trainer for auto encoder')
    parser.add_argument('--arch', default='vgg16', type=str, help='backbone architecture')
    parser.add_argument('--checkpoint', type=str, help='checkpoint of model to use (.pth file)')
    parser.add_argument('--dataset-name', type=str, help='name of dataset')
    parser.add_argument('--num-images', type=int, default=8, help='max number of images displayed')

    args = parser.parse_args()

    args.parallel = 0
    args.batch_size = 1
    args.workers = 0

    return args

def main(args):
    utils.init_seeds(1, cuda_deterministic=False)

    print('Modeling network')
    model = builder.BuildAutoEncoder(args)
    total_params = sum(p.numel() for p in model.parameters())
    print('Number of parameters: {}'.format(total_params))

    print('Loading parameters from {}'.format(args.checkpoint))
    utils.load_dict(args.checkpoint, model)

    print('Building dataloader')
    val_loader = dataloader.val_loader(args)

    print('Reconstructing')

    model.eval()
    num_images = args.num_images
    total_images = num_images * 2
    cols = math.ceil(math.sqrt(total_images))
    cols = cols if cols % 2 == 0 else cols+1
    rows = max(math.ceil(total_images / cols), 2)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    for col in range(cols):
        title = "Original" if col % 2 == 0 else "Reconstructed"
        axes[0, col].set_title(title)

    with torch.no_grad():
        for i, (input, target) in enumerate(tqdm(val_loader, total=total_images)):
            if i == total_images:
                break
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            
            reconstructed = model(input)

            input_img = transforms.ToPILImage()(input.squeeze().cpu())
            reconstructed_img = transforms.ToPILImage()(reconstructed.squeeze().cpu())

            row = i // cols
            col = (i % (cols // 2)) * 2
            axes[row, col].imshow(input_img)
            axes[row, col].axis('off')
            axes[row, col + 1].imshow(reconstructed_img)
            axes[row, col + 1].axis('off')


    for i in range(total_images, rows * cols):
            fig.delaxes(axes[i // cols, i % cols])
    plt.tight_layout()
    plt.savefig('figs/reconstruction.jpg')

if __name__ == '__main__':
    args = parse_args()
    main(args)
