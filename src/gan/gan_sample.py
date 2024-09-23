#!/usr/bin/env python

import argparse
import torch
from torch import nn
from torchvision.utils import save_image
import os
from torchvision import transforms
import numpy as np
from PIL import Image

def sample_and_assemble_images(generator, num_images, save_dir, device, debug, overlap, gen_type):
    generator.eval()
    
    # Determine patch dimensions
    patch_size = gen_type
    patches_per_row = 3072 // (patch_size - overlap)
    patches_per_col = 2048 // (patch_size - overlap)
    num_samples = patches_per_row * patches_per_col

    for n in range(num_images):
        if debug:
            noise = torch.randn(1, 100, 29, 45, device=device) 
        else:
            noise = torch.randn(num_samples, 100, 1, 1, device=device)
        
        with torch.no_grad():
            fake_images = generator(noise).detach().cpu()
        
            big_image = Image.new('RGB', (3072, 2048))
            
            for i in range(patches_per_row):
                for j in range(patches_per_col):
                    idx = i * patches_per_col + j
                    if idx < len(fake_images):
                        patch_image = transforms.ToPILImage()(fake_images[idx])
                        x = i * (patch_size - overlap)
                        y = j * (patch_size - overlap)
                        big_image.paste(patch_image, (x, y), patch_image.convert('RGBA'))
            save_path = os.path.join(save_dir, f'generated_image_{n}.png')
            big_image.save(save_path)
            print(f'Saved generated image {n+1} to {save_dir}')




class Generator128(nn.Module):
    def __init__(self):
        super(Generator128, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 64 * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 16, 64 * 8, 4, 2, 1, bias=False),
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

class Generator64(nn.Module):
    def __init__(self):
        super(Generator64, self).__init__()
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

    if args.generator == 64:
        netG = Generator64().to(device)
    if args.generator == 128:
        netG = Generator128().to(device)
    netG.load_state_dict(checkpoint['generator_state_dict'])

    os.makedirs(args.save_dir, exist_ok=True)
    sample_and_assemble_images(netG, num_images=args.num_images, save_dir=args.save_dir, device=device, debug=args.debug, overlap=args.overlap, gen_type=args.generator)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate large images from a GAN generator")
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the checkpoint file')
    parser.add_argument('--num_images', type=int, required=True, help='Number of large images to generate')
    parser.add_argument('--generator', type=int, required=True, choices=[64, 128], help='Generator for 64x64 images or 128x128 images')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save generated images')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for computation (default: cpu)')
    parser.add_argument('--overlap', type=int, default='0', help='Amount of overlap to apply between patches of the reconstructed image (Default 0)')

    args = parser.parse_args()
    main(args)