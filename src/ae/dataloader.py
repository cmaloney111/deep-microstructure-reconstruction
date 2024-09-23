#!/usr/bin/env python

from PIL import Image
import torch
import torch.utils.data as data
from torchvision.transforms import transforms

class ImageDataset(data.Dataset):
    def __init__(self, dataset_name, transform=None):
        self.transform = transform
        self.im_names = []
        with open("list/" + dataset_name + "_list.txt", 'r') as f:
            lines = f.readlines()
            for line in lines:
                data = line.strip().split(' ')
                self.im_names.append(data[0])

    def __getitem__(self, index):
        im_name = self.im_names[index]
        img = Image.open(im_name).convert('RGB') 
        img = self.transform(img)
        return img, img

    def __len__(self):
        return len(self.im_names)

def train_loader(args):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    train_dataset = ImageDataset(args.dataset_name, transform=train_transform)   
    if args.parallel == 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
                            train_dataset,
                            rank=args.rank,
                            num_replicas=args.world_size,
                            shuffle=True)         
    else:  
        train_sampler = None    
    train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    shuffle=(train_sampler is None),
                    batch_size=args.batch_size,
                    num_workers=args.workers,
                    pin_memory=True,
                    sampler=train_sampler,
                    drop_last=(train_sampler is None))
    return train_loader

def val_loader(args):
    val_transform = transforms.Compose([
                        transforms.Resize(256),                   
                        transforms.CenterCrop(224),
                        transforms.ToTensor()
                    ])
    val_dataset = ImageDataset(args.dataset_name, transform=val_transform)      
    val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    shuffle=False,
                    batch_size=args.batch_size,
                    num_workers=args.workers,
                    pin_memory=True)
    return val_loader