#!/usr/bin/env python

import argparse
import math
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import transforms

import os
from tqdm import tqdm
import numpy as np

import sys
sys.path.append(".")

import utils
import models.builder as builder

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_args():
    print('Parsing arguments')
    parser = argparse.ArgumentParser(description='Decode latent features into images')
    parser.add_argument('--arch', default='vgg19', type=str, help='backbone architechture (default vgg19)')
    parser.add_argument('--checkpoint', type=str, help='checkpoint of model to use (.pth file)')
    parser.add_argument('--num-images', type=int, default=128, help='max number of images to display (default 128)')
    parser.add_argument('--array-path', type=str, help='path to a .npy file containing an array')
    parser.add_argument('--array-folder', type=str, help='path to a folder containing .npy files')
    parser.add_argument('--output-path', default='figs/generation.jpg', type=str,
                        help='file to output the decoded images to (default figs/generation.jpg)')
    parser.add_argument('--interpolate', action='store_true',
                         help='perform interpolation between two latent vectors')
    parser.add_argument('--num-steps', type=int, default=16, help='number of interpolation steps (default 16)')
    
    args = parser.parse_args()

    if args.array_path and args.array_folder:
        raise ValueError("Both --array-path and --array-folder cannot be given at the same time")

    args.parallel = 0
    args.batch_size = 1
    args.workers = 0

    return args

def random_sample(arch):
    if arch in ["vgg11", "vgg13", "vgg16", "vgg19", "resnet18", "resnet34"]:
        return torch.randn((1,512,7,7))
    elif arch in ["resnet50", "resnet101", "resnet152"]:
        return torch.randn((1,2048,7,7))
    else:
        raise NotImplementedError("Invalid architecture given")

def interpolate_vectors(v1, v2, num_steps):
    alphas = np.linspace(0, 1, num_steps)
    interpolated_vectors = [(1 - alpha) * v1 + alpha * v2 for alpha in alphas]
    return torch.stack(interpolated_vectors)

def display_images(images, num_images, output_path):
    rows = math.ceil(math.sqrt(num_images))
    cols = math.ceil(num_images / rows)

    plt.figure(figsize=(cols * 2, rows * 2))

    for i in range(num_images):
        plt.subplot(rows, cols, i + 1, xticks=[], yticks=[])
        plt.imshow(images[i])

    plt.tight_layout()
    plt.savefig(output_path)

def main(args):
    utils.init_seeds(1, cuda_deterministic=False)

    print('Modeling network')
    model = builder.BuildAutoEncoder(args)
    total_params = sum(p.numel() for p in model.parameters())
    print('Number of parameters: {}'.format(total_params))

    print('Loading parameters from {}'.format(args.checkpoint))
    utils.load_dict(args.checkpoint, model)

    transform = transforms.ToPILImage()

    print('Generating image(s)')

    if args.array_path:
        input = torch.from_numpy(np.load(args.array_path)).cuda()
        output = model.module.decoder(input)
        images = [transform(output.squeeze().cpu())]
    elif args.array_folder:
        images = []
        array_folder = os.listdir(args.array_folder)
        num_images = min(args.num_images, len(array_folder))
        
        if args.interpolate:
            file1, file2 = np.random.choice(array_folder, 2, replace=False)
            latent_vector1 = torch.from_numpy(np.load(os.path.join(args.array_folder, file1))).cuda()
            latent_vector2 = torch.from_numpy(np.load(os.path.join(args.array_folder, file2))).cuda()

            interpolated_latent_vectors = interpolate_vectors(latent_vector1, latent_vector2, args.num_steps)

            for latent_vector in tqdm(interpolated_latent_vectors, total=args.num_steps):
                output = model.module.decoder(latent_vector)
                images.append(transform(output.squeeze().cpu()))
        else:
            for i, file in enumerate(tqdm(array_folder, total=num_images)):
                if i == num_images:
                    break
                input = torch.from_numpy(np.load(os.path.join(args.array_folder, file))).cuda()
                output = model.module.decoder(input)
                images.append(transform(output.squeeze().cpu()))
    else:
        num_images = args.num_images
        model.eval()
        images = []
        with torch.no_grad():
            for _ in tqdm(range(num_images)):
                input = random_sample(arch=args.arch).cuda()
                output = model.module.decoder(input)
                images.append(transform(output.squeeze().cpu()))

    display_images(images, len(images), args.output_path)

if __name__ == '__main__':
    args = parse_args()
    main(args)
