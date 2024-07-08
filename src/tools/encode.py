#!/usr/bin/env python

import argparse
from PIL import Image
import torch
from torchvision.transforms import transforms
import numpy as np
import os
from tqdm import tqdm
import sys
sys.path.append("./")
import utils
import models.builder as builder

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_args():
    print('Parsing argumments')
    parser = argparse.ArgumentParser(description='Encoder for images')
    parser.add_argument('--arch', default='vgg19', type=str, help='backbone architechture (default vgg19)')
    parser.add_argument('--checkpoint', type=str, help='checkpoint of model to use (.pth file)')
    parser.add_argument('--img-path',type=str, help='path of image to be encoded')
    parser.add_argument('--output-path',type=str, required=True, 
                        help='path to output encoded array(s) - file if img-path given and folder if img-folder given')
    parser.add_argument('--img-folder',type=str, help='path of folder that contains images to be encoded')              
    parser.add_argument('--num-images',type=int, default=128, help='max number of images to be encoded (default 128)')
    
    args = parser.parse_args()

    if args.img_path and args.img_folder:
        raise ValueError("Both --img-path and --img-folder cannot be given at the same time")
    elif not args.img_path and not args.img_folder:
        raise ValueError("One of either --img-path or --img-folder must be given")

    args.parallel = 0
    args.batch_size = 1
    args.workers = 0
    return args

def encode(model, trans, img_path):
    img = Image.open(img_path).convert("RGB")
    img = trans(img).unsqueeze(0).cuda()
    model.eval()
    with torch.no_grad():
        code = model.module.encoder(img).cpu().numpy()
    return code

def main(args):
    utils.init_seeds(1, cuda_deterministic=False)

    print('Modeling network')
    model = builder.BuildAutoEncoder(args)     
    total_params = sum(p.numel() for p in model.parameters())
    print('Number of parameters: {}'.format(total_params))

    print('Loading parameters from {}'.format(args.checkpoint))
    utils.load_dict(args.checkpoint, model)
    
    transform = transforms.Compose([
                    transforms.Resize(256),                   
                    transforms.CenterCrop(224),
                    transforms.ToTensor()
                  ])

    if args.img_path:
        code = encode(model, transform, args.img_path)
        np.save(args.output_path, code)
    else:    
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)
        img_folder = os.listdir(args.img_folder)
        num_images = min(args.num_images, len(img_folder))
        for i, filename in enumerate(tqdm(img_folder, total=num_images)):
            if i == num_images:
                break
            code = encode(model, transform, os.path.join(args.img_folder, filename))
            output_file = os.path.join(args.output_path, "encoded_{}{}".format(filename[:filename.rfind('.')], '.npy'))
            np.save(output_file, code)
            

if __name__ == '__main__':

    args = parse_args()

    main(args)


