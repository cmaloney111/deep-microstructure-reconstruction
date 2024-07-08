#!/usr/bin/env python

import torch
from torch import nn
from torchvision.utils import save_image
import os

def sample_images(generator, num_samples, save_dir, device):
    generator.eval()
    noise = torch.randn(num_samples, 100, 1, 1, device=device)
    with torch.no_grad():
        fake_images = generator(noise).detach().cpu()
    for i in range(num_samples):
        save_image(fake_images[i], os.path.join(save_dir, f'generated_image_{i}.png'))
    print(f'Saved {num_samples} generated images to {save_dir}')

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


checkpoint = torch.load(r'results\gan\checkpoint_epoch_1700.pth')
device = torch.device('cpu')

netG = Generator().to(device)
netG.load_state_dict(checkpoint['generator_state_dict'])

sample_images(netG, num_samples=5, save_dir='figs/generated_images', device=device)