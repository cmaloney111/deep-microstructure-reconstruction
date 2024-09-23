import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision

class MicrostructureDataset(Dataset):
    def __init__(self, txt_file, transform=None, crop_size=512):
        with open(txt_file, 'r') as file:
            self.image_paths = [line.split()[0] for line in file.readlines()]
        self.transform = transform
        self.crop_size = crop_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')
        image = np.array(image)
        h, w = image.shape
        top = np.random.randint(0, h - self.crop_size)
        left = np.random.randint(0, w - self.crop_size)
        image = image[top: top + self.crop_size, left: left + self.crop_size]
        if self.transform:
            image = self.transform(image)
        return image

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomCrop(512),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = MicrostructureDataset(txt_file='MR_bin_jpg_list.txt', transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

class Generator(nn.Module):
    def __init__(self, z_dim=100, img_channels=1):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            self._block(z_dim, 1024, 4, 1, 0), 
            self._block(1024, 512, 4, 2, 1),   
            self._block(512, 256, 4, 2, 1),    
            self._block(256, 128, 4, 2, 1),    
            self._block(128, 64, 4, 2, 1),     
            self._block(64, 32, 4, 2, 1),      
            self._block(32, 16, 4, 2, 1),      
            nn.ConvTranspose2d(16, img_channels, kernel_size=4, stride=2, padding=1), # output: img_channels x 512 x 512
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.gen(x)


class Discriminator(nn.Module):
    def __init__(self, img_channels=1):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=4, stride=2, padding=1), 
            nn.LeakyReLU(0.2, inplace=True),
            self._block(64, 128, 4, 2, 1),   
            self._block(128, 256, 4, 2, 1),  
            self._block(256, 512, 4, 2, 1),  
            self._block(512, 1024, 4, 2, 1), 
            self._block(1024, 2048, 4, 2, 1),
            self._block(2048, 4096, 4, 2, 1),
            nn.Conv2d(4096, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid(),
        )
        
        self.first = nn.Conv2d(img_channels, 64, kernel_size=4, stride=2, padding=1)
        self.second = nn.LeakyReLU(0.2, inplace=True)


        self.third = self._block(64, 128, 4, 2, 1)
        self.a = self._block(128, 256, 4, 2, 1)
        self.b = self._block(256, 512, 4, 2, 1)
        self.c = self._block(512, 1024, 4, 2, 1)
        self.d = self._block(1024, 2048, 4, 2, 1)
        self.e = self._block(2048, 4096, 4, 2, 1)
        self.f = nn.Conv2d(4096, 1, kernel_size=4, stride=1, padding=0)
        self.g = nn.Sigmoid()
        

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.disc(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 2e-4
z_dim = 100
image_channels = 1
batch_size = 8
num_epochs = 2000

gen = Generator(z_dim, image_channels).to(device)
disc = Discriminator(image_channels).to(device)

opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))

criterion = nn.BCELoss()
fixed_noise = torch.randn(batch_size, z_dim, 1, 1).to(device)

gen.train()
disc.train()

for epoch in range(num_epochs):
    gen_losses = []
    disc_losses = []

    for batch in dataloader:
        real_images = batch.to(device)
        batch_size = real_images.size(0)

        disc.zero_grad()
        noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
        fake_images = gen(noise)
        disc_fake_pred = disc(fake_images.detach())
        disc_real_pred = disc(real_images)
        disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
        disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
        disc_loss = (disc_fake_loss + disc_real_loss) / 2
        disc_loss.backward()
        opt_disc.step()
        disc_losses.append(disc_loss.item())

        gen.zero_grad()
        disc_fake_pred = disc(fake_images)
        gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
        gen_loss.backward()
        opt_gen.step()
        gen_losses.append(gen_loss.item())

    avg_gen_loss = sum(gen_losses) / len(gen_losses)
    avg_disc_loss = sum(disc_losses) / len(disc_losses)
    print(f"Epoch [{epoch}/{num_epochs}] Generator Loss: {avg_gen_loss:.4f}, Discriminator Loss: {avg_disc_loss:.4f}")
    if (epoch - 1) % 300 == 0:
          checkpoint_path = f'checkpoint/checkpoint_epoch_{epoch - 1}.pth'
          torch.save({
              'epoch': epoch,
              'gen_state_dict': gen.state_dict(),
              'disc_state_dict': disc.state_dict(),
              'opt_gen_state_dict': opt_gen.state_dict(),
              'opt_disc_state_dict': opt_disc.state_dict(),
              'gen_loss': avg_gen_loss,
              'disc_loss': avg_disc_loss,
          }, checkpoint_path)
          print(f"Checkpoint saved at {checkpoint_path}")
    with open('log.txt', 'a') as f:
        f.write(f"Epoch [{epoch}/{num_epochs}] Generator Loss: {avg_gen_loss:.4f}, Discriminator Loss: {avg_disc_loss:.4f}\n")
