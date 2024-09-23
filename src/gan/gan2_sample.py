#!/usr/bin/env python

import argparse
import torch
from torch import nn
from torchvision.utils import save_image
import os
from torchvision import transforms
import numpy as np
from PIL import Image

def sample_and_assemble_images(generator, num_images, save_dir, device, debug):
    generator.eval()
    patch_size = 128  # Updated patch size
    num_samples = (3072 // patch_size) * (2048 // patch_size)
    
    for n in range(num_images):    
        noise = torch.randn(num_samples, 100, 1, 1, device=device)
        with torch.no_grad():
            fake_images = generator(noise).detach().cpu()
        big_image = Image.new('L', (3072, 2048))
        for i in range(3072 // patch_size):
            for j in range(2048 // patch_size):
                idx = i * (2048 // patch_size) + j
                img = transforms.ToPILImage()(fake_images[idx])
                big_image.paste(img, (i * patch_size, j * patch_size))
            if debug:
                print("Loaded image {}".format(i))
        
        save_path = os.path.join(save_dir, f'generated_image_{n}.png')
        big_image.save(save_path)
        print(f'Saved generated image {n+1} to {save_dir}')

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

def main(args):
    checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
    device = torch.device(args.device)

    netG = Generator().to(device)
    netG.load_state_dict(checkpoint['generator_state_dict'])

    os.makedirs(args.save_dir, exist_ok=True)
    sample_and_assemble_images(netG, num_images=args.num_images, save_dir=args.save_dir, device=device, debug=args.debug)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate large images from a GAN generator")
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the checkpoint file')
    parser.add_argument('--num_images', type=int, required=True, help='Number of large images to generate')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save generated images')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for computation (default: cpu)')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    main(args)
