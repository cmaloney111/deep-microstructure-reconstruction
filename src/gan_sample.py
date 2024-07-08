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


checkpoint = torch.load(r'results\gan\checkpoint_epoch_90.pth')
device = torch.device('cpu')

netG = Generator().to(device)
netG.load_state_dict(checkpoint['generator_state_dict'])

sample_images(netG, num_samples=5, save_dir='figs/generated_images', device=device)







import os
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import porespy as ps

def sample_images(generator, num_samples, save_dir, device):
    generator.eval()
    noise = torch.randn(num_samples, 100, 1, 1, device=device)
    with torch.no_grad():
        fake_images = generator(noise).detach().cpu()
    for i in range(num_samples):
        save_image(fake_images[i], os.path.join(save_dir, f'generated_image_{i}.png'))
    print(f'Saved {num_samples} generated images to {save_dir}')

class MicrostructureDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.image_paths = []
        self.labels = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                path, label = line.strip().split(' ')
                self.image_paths.append(path)
                self.labels.append(int(label))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

dataset = MicrostructureDataset('list/MR_bin_jpg_list.txt', transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
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

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.a = nn.Conv2d(1, 64, 4, 2, 1, bias=False)
        self.b = nn.LeakyReLU(0.2, inplace=True)
        self.c = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.d= nn.BatchNorm2d(128)
        self.e= nn.LeakyReLU(0.2, inplace=True)
        self.f= nn.Conv2d(128, 256, 4, 2, 1, bias=False)
        self.g= nn.BatchNorm2d(256)
        self.h= nn.LeakyReLU(0.2, inplace=True)
        self.i= nn.Conv2d(256, 512, 4, 2, 1, bias=False)
        self.j= nn.BatchNorm2d(512)
        self.k= nn.LeakyReLU(0.2, inplace=True)
        self.l= nn.Conv2d(512, 1, 4, 1, 0, bias=False)
        self.m= nn.Sigmoid()
        

    def forward(self, input):
        a = self.a(input)
        print(a.shape)
        b = self.b(a)
        print(b.shape)
        c = self.c(b)
        print(c.shape)
        d = self.d(c)
        print(d.shape)
        e = self.e(d)
        print(e.shape)
        f = self.f(e)
        print(f.shape)
        g = self.g(f)
        print(g.shape)
        h = self.h(g)
        print(h.shape)
        i = self.i(h)
        print(i.shape)
        j = self.j(i)
        print(j.shape)
        k = self.k(j)
        print(k.shape)
        l = self.l(k)
        print(l.shape)
        m = self.m(l)
        print(m.shape)
        return self.main(input)

netG = Generator().to(device)
netD = Discriminator().to(device)

criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

def lineal_path_function(image):
    return ps.metrics.lineal_path_distribution(image)

def chord_length_distribution(image):
    return ps.metrics.chord_length_distribution(image)

def pore_size_distribution(image):
    return ps.metrics.pore_size_distribution(image)

def two_point_correlation(image):
    return ps.metrics.two_point_correlation(image)

num_epochs = 2
fixed_noise = torch.randn(64, 100, 1, 1, device=device)

for epoch in range(num_epochs):
    for i, (images, _) in enumerate(dataloader):
        netD.zero_grad()
        real_images = images.to(device)
        b_size = real_images.size(0)
        label = torch.full((b_size,), 1., device=device)
        print(label.shape)
        output = netD(real_images).view(-1)
        print(output.shape)
        errD_real = criterion(output, label)
        errD_real.backward()
        
        noise = torch.randn(b_size, 100, 1, 1, device=device)
        fake_images = netG(noise)
        label.fill_(0.)
        output = netD(fake_images.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        optimizerD.step()
        
        netG.zero_grad()
        label.fill_(1.)
        output = netD(fake_images).view(-1)
        errG = criterion(output, label)
        errG.backward()
        optimizerG.step()
        
        if i % 50 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}] Batch [{i}/{len(dataloader)}] Loss D: {errD_real + errD_fake}, Loss G: {errG}')
    
    if epoch % 10 == 0:
        torch.save({
            'epoch': epoch,
            'generator_state_dict': netG.state_dict(),
            'discriminator_state_dict': netD.state_dict(),
            'optimizerG_state_dict': optimizerG.state_dict(),
            'optimizerD_state_dict': optimizerD.state_dict(),
        }, f'results/gan/checkpoint_epoch_{epoch}.pth')

    with torch.no_grad():
        fake_images = netG(fixed_noise).detach().cpu()
        
    real_image = images[0].cpu().numpy()
    fake_image = fake_images[0].cpu().numpy()

    lpf_real = lineal_path_function(real_image[0])
    lpf_fake = lineal_path_function(fake_image[0])
    
    cld_real = chord_length_distribution(real_image[0])
    cld_fake = chord_length_distribution(fake_image[0])
    
    psd_real = pore_size_distribution(real_image[0])
    psd_fake = pore_size_distribution(fake_image[0])
    
    tpc_real = two_point_correlation(real_image[0])
    tpc_fake = two_point_correlation(fake_image[0])

    if epoch % 10 == 0:
        print(f'+------------------------+-------------------+-------------------+')
        print(f'| Metric                 | Real              | Fake              |')
        print(f'+------------------------+-------------------+-------------------+')
        print(f'| Lineal Path Function   | {np.mean(lpf_real.L):.6f}       | {np.mean(lpf_fake.L):.6f}       |')
        print(f'| Chord Length Dist.     | {np.mean(cld_real.L):.6f}       | {np.mean(cld_fake.L):.6f}       |')
        print(f'| Pore Size Distribution | {np.mean(psd_real.bin_centers):.6f}       | {np.mean(psd_fake.bin_centers):.6f}       |')
        print(f'| Two-Point Correlation  | {np.mean(tpc_real.probability):.6f}       | {np.mean(tpc_fake.probability):.6f}       |')
        print(f'+------------------------+-------------------+-------------------+')
