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
    num_samples = 1  # 3072x2048 / 64x64 = 48x32 = 1536
    
    for n in range(num_images):    
        # noise = torch.randn(1, 100, 110, 74, device=device)
        noise = torch.randn(num_samples, 128, 29, 45, device=device)
        with torch.no_grad():
            fake_images = generator(noise).detach().cpu()
        print(fake_images.shape)
        fake_images = fake_images.squeeze(0)
        img = transforms.ToPILImage()(fake_images)
        save_path = os.path.join(save_dir, f'generated_image_{n}.png')
        img.save(save_path)
        
        # big_image = Image.new('L', (3072, 2048))
        # for i in range(48):  # 3072 / 64 = 48
        #     for j in range(32):  # 2048 / 64 = 32
        #         idx = i * 32 + j
        #         img = transforms.ToPILImage()(fake_images[idx])
        #         big_image.paste(img, (i * 64, j * 64))
        #     if debug:
        #         print("Loaded image {}".format(i))
        
        # save_path = os.path.join(save_dir, f'generated_image_{n}.png')
        # big_image.save(save_path)
        # print(f'Saved generated image {n+1} to {save_dir}')

class Generator(torch.nn.Module):
		def __init__(self):
			super(Generator, self).__init__()
			main = torch.nn.Sequential()

			# We need to know how many layers we will use at the beginning
			mult = 256 // 8

			### Start block
			# Z_size random numbers
			main.add_module('Start-ConvTranspose2d', torch.nn.ConvTranspose2d(128, 128 * mult, kernel_size=4, stride=1, padding=0, bias=False))
			main.add_module('Start-BatchNorm2d', torch.nn.BatchNorm2d(128 * mult))
			main.add_module('Start-ReLU', torch.nn.ReLU())
			# Size = (G_h_size * mult) x 4 x 4

			### Middle block (Done until we reach ? x image_size/2 x image_size/2)
			i = 1
			while mult > 1:
				main.add_module('Middle-ConvTranspose2d [%d]' % i, torch.nn.ConvTranspose2d(128 * mult, 128 * (mult//2), kernel_size=4, stride=2, padding=1, bias=False))
				main.add_module('Middle-BatchNorm2d [%d]' % i, torch.nn.BatchNorm2d(128 * (mult//2)))
				main.add_module('Middle-ReLU [%d]' % i, torch.nn.ReLU())
				# Size = (G_h_size * (mult/(2*i))) x 8 x 8
				mult = mult // 2
				i += 1

			main.add_module('End-ConvTranspose2d', torch.nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1, bias=False))
			main.add_module('End-Tanh', torch.nn.Tanh())
			# Size = n_colors x image_size x image_size
			self.main = main

		def forward(self, input):
			output = self.main(input)
			return output

def main(args):
    checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
    device = torch.device(args.device)

    netG = Generator().to(device)
    netG.load_state_dict(checkpoint['G_state'])

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