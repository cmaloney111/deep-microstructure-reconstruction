import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision import utils as vutils

import argparse
import random
import os
from tqdm import tqdm

from PIL import Image

from models_cond import weights_init, Discriminator, Generator, calculate_porosity
from operation_cond import copy_G_params, load_params, get_dir
from operation_cond import ImageFolder, ClassConditionedDataset, InfiniteSamplerWrapper, calculate_image_porosity
from diffaug import DiffAugment
policy = 'color,translation'
import lpips

use_gpu = torch.cuda.is_available()
percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=use_gpu)


def crop_image_by_part(image, part):
    hw = image.shape[2]//2
    if part==0:
        return image[:,:,:hw,:hw]
    if part==1:
        return image[:,:,:hw,hw:]
    if part==2:
        return image[:,:,hw:,:hw]
    if part==3:
        return image[:,:,hw:,hw:]


def calculate_class_porosities(data_root, im_size, device):
    """
    Calculate the average porosity for each class in the dataset
    
    Args:
        data_root: Path to the dataset root directory
        im_size: Image size for processing
        device: Device to use for calculation
        
    Returns:
        Dictionary mapping class indices to average porosity values
    """
    print("Calculating average porosity for each class...")
    
    # Find all class directories
    classes = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    
    # Transform for loading images
    transform = transforms.Compose([
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Calculate porosity for each class
    class_porosities = {}
    
    for class_name in tqdm(classes, desc="Classes"):
        class_dir = os.path.join(data_root, class_name)
        class_idx = class_to_idx[class_name]
        
        # Get all image files in this class directory
        image_files = []
        for file in os.listdir(class_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                image_files.append(os.path.join(class_dir, file))
        
        if not image_files:
            print(f"Warning: No images found in class {class_name}")
            continue
            
        # Calculate porosity for each image in this class
        porosities = []
        
        for img_path in tqdm(image_files, desc=f"Class {class_name}", leave=False):
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension
                porosity = calculate_porosity(img_tensor)
                porosities.append(porosity[0].item())  # Take first element since we added batch dimension
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        # Calculate average porosity for this class
        if porosities:
            avg_porosity = sum(porosities) / len(porosities)
            class_porosities[class_idx] = avg_porosity
            print(f"Class {class_name} (index {class_idx}): Average porosity = {avg_porosity:.4f}")
    
    # Save porosity values to disk
    save_path = os.path.join(os.path.dirname(data_root), 'porosity_values.pt')
    torch.save(class_porosities, save_path)
    print(f"Porosity values saved to {save_path}")
    
    return class_porosities


def train_d(net, data, label="real", part=None, labels=None):
    """Train function of discriminator"""
    if label=="real":
        if part is None:
            part = random.randint(0, 3)
        pred, [rec_all, rec_small, rec_part] = net(data, label, part=part, labels=labels)
        
        # Basic loss
        err = F.relu(torch.rand_like(pred) * 0.2 + 0.8 - pred).mean()
        
        # Reconstruction loss
        err += percept(rec_all, F.interpolate(data, rec_all.shape[2])).sum()
        err += percept(rec_small, F.interpolate(data, rec_small.shape[2])).sum()
        err += percept(rec_part, F.interpolate(crop_image_by_part(data, part), rec_part.shape[2])).sum()
        
        # Porosity loss
        if labels is not None:
            real_porosity = calculate_porosity(data)
            fake_porosity = calculate_porosity(rec_all)
            porosity_loss = F.mse_loss(fake_porosity, real_porosity)
            err += porosity_loss * 10.0  # Weight for porosity loss
        
        err.backward()
        return pred.mean().item(), rec_all, rec_small, rec_part
    else:
        pred = net(data, label, labels=labels)
        err = F.relu(torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()
        err.backward()
        return pred.mean().item()


def train(args):
    data_root = args.path
    total_iterations = args.iter
    checkpoint = args.ckpt
    batch_size = args.batch_size
    im_size = args.im_size
    ndf = 64
    ngf = 64
    nz = 256
    nlr = 0.0002
    nbeta1 = 0.5
    use_cuda = use_gpu
    multi_gpu = args.multi_gpu
    dataloader_workers = args.workers
    current_iteration = args.start_iter
    save_interval = args.save_interval
    saved_model_folder, saved_image_folder = get_dir(args)
    
    # Check if the data directory has subdirectories (class folders)
    is_conditional = False
    num_classes = 0
    embedding_dim = 100  # Dimension for class embedding
    
    if os.path.isdir(os.path.join(data_root, next(os.walk(data_root))[1][0])):
        is_conditional = True
        class_dirs = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
        num_classes = len(class_dirs)
        print(f"Running in conditional mode with {num_classes} classes: {class_dirs}")

    device = torch.device("cpu")
    if use_cuda:
        device = torch.device(f"cuda:{args.cuda}")

    print(f"Using device: {device}")
    # Calculate or load porosity targets if in conditional mode
    class_porosities = None
    if is_conditional:
        porosity_file = os.path.join(os.path.dirname(data_root), 'porosity_values.pt')
        if os.path.exists(porosity_file) and not args.recalculate_porosity:
            # Load pre-calculated porosity values
            class_porosities = torch.load(porosity_file)
            print(f"Loaded pre-calculated porosity values from {porosity_file}")
            for class_idx, porosity in class_porosities.items():
                print(f"Class {class_idx}: Target porosity = {porosity:.4f}")
        else:
            # Calculate porosity values for each class
            class_porosities = calculate_class_porosities(data_root, im_size, device)

    transform_list = [
            transforms.Resize((int(im_size), int(im_size))),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
    trans = transforms.Compose(transform_list)
    
    if 'lmdb' in data_root:
        from operation_cond import MultiResolutionConditionalDataset, MultiResolutionDataset
        if is_conditional:
            dataset = MultiResolutionConditionalDataset(data_root, trans, im_size)
        else:
            dataset = MultiResolutionDataset(data_root, trans, im_size)
    else:
        if is_conditional:
            dataset = ClassConditionedDataset(root=data_root, transform=trans)
        else:
            dataset = ImageFolder(root=data_root, transform=trans)

    # Create dataloader
    if is_conditional:
        dataloader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            sampler=InfiniteSamplerWrapper(dataset), num_workers=dataloader_workers))
    else:
        dataloader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            sampler=InfiniteSamplerWrapper(dataset), num_workers=dataloader_workers))
    
    # Create generator and discriminator models
    netG = Generator(ngf=ngf, nz=nz, im_size=im_size, num_classes=num_classes, embedding_dim=embedding_dim)
    netG.apply(weights_init)

    netD = Discriminator(ndf=ndf, im_size=im_size, num_classes=num_classes, embedding_dim=embedding_dim)
    netD.apply(weights_init)

    netG.to(device)
    netD.to(device)

    avg_param_G = copy_G_params(netG)

    # Create fixed noise for evaluation
    fixed_noise = torch.FloatTensor(8, nz).normal_(0, 1).to(device)
    
    # For conditional GAN, also create fixed labels for evaluation
    if is_conditional:
        fixed_labels = torch.arange(0, min(8, num_classes), dtype=torch.long).to(device)
        # Repeat labels if needed to match batch size
        if len(fixed_labels) < 8:
            fixed_labels = fixed_labels.repeat(8 // len(fixed_labels) + 1)[:8]
    
    # Setup optimizers
    optimizerG = optim.Adam(netG.parameters(), lr=nlr, betas=(nbeta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=nlr, betas=(nbeta1, 0.999))

    # Load checkpoint if provided
    if checkpoint != 'None':
        ckpt = torch.load(checkpoint)
        netG.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['g'].items()})
        netD.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['d'].items()})
        avg_param_G = ckpt['g_ema']
        optimizerG.load_state_dict(ckpt['opt_g'])
        optimizerD.load_state_dict(ckpt['opt_d'])
        current_iteration = int(checkpoint.split('_')[-1].split('.')[0])
        del ckpt
    
    # Set up for multi-GPU training if enabled
    if multi_gpu and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        netG = nn.DataParallel(netG.to(device))
        netD = nn.DataParallel(netD.to(device))
    
    # Main training loop
    for iteration in tqdm(range(current_iteration, total_iterations+1)):
        # Get real data
        if is_conditional:
            real_data = next(dataloader)
            real_image, real_labels = real_data
            real_image = real_image.to(device)
            real_labels = real_labels.to(device)
        else:
            real_image = next(dataloader)
            real_image = real_image.to(device)
            real_labels = None
            
        current_batch_size = real_image.size(0)
        
        # Generate random noise
        noise = torch.Tensor(current_batch_size, nz).normal_(0, 1).to(device)
        
        # For conditional GAN, randomly generate labels for fake images
        if is_conditional:
            fake_labels = torch.randint(0, num_classes, (current_batch_size,), dtype=torch.long).to(device)
            fake_images = netG(noise, fake_labels)
        else:
            fake_images = netG(noise)

        # Apply DiffAugment
        real_image = DiffAugment(real_image, policy=policy)
        fake_images = [DiffAugment(fake, policy=policy) for fake in fake_images]
        
        # 1. Train Discriminator
        netD.zero_grad()

        err_dr, rec_img_all, rec_img_small, rec_img_part = train_d(netD, real_image, label="real", labels=real_labels)
        
        if is_conditional:
            train_d(netD, [fi.detach() for fi in fake_images], label="fake", labels=fake_labels)
        else:
            train_d(netD, [fi.detach() for fi in fake_images], label="fake")
            
        optimizerD.step()
        
        # 2. Train Generator
        netG.zero_grad()
        
        if is_conditional:
            pred_g = netD(fake_images, "fake", labels=fake_labels)
        else:
            pred_g = netD(fake_images, "fake")
            
        err_g = -pred_g.mean()
        
        # Add porosity loss for generator with pre-calculated targets
        if is_conditional and class_porosities is not None:
            # Get pre-calculated target porosity for each class in the batch
            target_porosity = []
            for label in fake_labels:
                label_idx = label.item()
                if label_idx in class_porosities:
                    target_p = class_porosities[label_idx]
                else:
                    # Fallback if porosity not calculated
                    target_p = 0.5
                target_porosity.append(target_p)
            target_porosity = torch.tensor(target_porosity, device=device)
            
            # Calculate porosity of generated images
            gen_porosity = calculate_porosity(fake_images[0])
            
            # Add porosity loss to generator loss
            porosity_loss = F.mse_loss(gen_porosity, target_porosity)
            err_g += porosity_loss * 5.0  # Weight for porosity loss

        err_g.backward()
        optimizerG.step()

        # Update moving average of G parameters
        for p, avg_p in zip(netG.parameters(), avg_param_G):
            avg_p.mul_(0.999).add_(0.001 * p.data)

        # Print progress
        if iteration % 100 == 0:
            print(f"GAN: loss d: {err_dr:.5f}    loss g: {-err_g.item():.5f}")
          
        # Save sample images
        if iteration % (save_interval*25) == 0 or iteration == 1:
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            with torch.no_grad():
                if is_conditional:
                    # Create a mapping from class indices to class names
                    class_names = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
                    class_names.sort()  # Sort to match the index order
                    
                    # Generate samples for each class
                    for class_idx in range(num_classes):
                        # Get the actual class name
                        class_name = class_names[class_idx]
                        
                        # Create class-specific labels tensor filled with this class index
                        class_labels = torch.full((8,), class_idx, dtype=torch.long, device=device)
                        
                        # Generate images with fixed noise and this class
                        class_images = netG(fixed_noise, class_labels)[0].add(1).mul(0.5)
                        
                        # Save class-specific samples with class name
                        vutils.save_image(class_images, 
                                        saved_image_folder+'/class_%s_iter_%d.jpg'%(class_name, iteration),
                                        nrow=4)
                    
                    # Also save the original mixed-class samples for backward compatibility
                    vutils.save_image(netG(fixed_noise, fixed_labels)[0].add(1).mul(0.5), 
                                    saved_image_folder+'/%d.jpg'%iteration, nrow=4)
                else:
                    vutils.save_image(netG(fixed_noise)[0].add(1).mul(0.5), 
                                    saved_image_folder+'/%d.jpg'%iteration, nrow=4)
                
                vutils.save_image(torch.cat([
                        F.interpolate(real_image, 128), 
                        rec_img_all, rec_img_small,
                        rec_img_part]).add(1).mul(0.5), saved_image_folder+'/rec_%d.jpg'%iteration)
            load_params(netG, backup_para)

        # Save model checkpoints
        if iteration % (save_interval*50) == 0 or iteration == total_iterations or iteration == 1:
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            torch.save({'g':netG.state_dict(),'d':netD.state_dict()}, saved_model_folder+'/%d.pth'%iteration)
            load_params(netG, backup_para)
            torch.save({'g':netG.state_dict(),
                        'd':netD.state_dict(),
                        'g_ema': avg_param_G,
                        'opt_g': optimizerG.state_dict(),
                        'opt_d': optimizerD.state_dict()}, saved_model_folder+'/all_%d.pth'%iteration)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Microstructure GAN')

    parser.add_argument('--path', type=str, default='../lmdbs/art_landscape_1k', help='path of resource dataset, should be a folder that has one or many sub image folders inside')
    parser.add_argument('--output_path', type=str, default='./', help='Output path for the train results')
    parser.add_argument('--cuda', type=int, default=0, help='index of gpu to use')
    parser.add_argument('--multi_gpu', action='store_true', default=False, help='use multiple gpus')
    parser.add_argument('--name', type=str, default='test1', help='experiment name')
    parser.add_argument('--iter', type=int, default=150000, help='number of iterations')
    parser.add_argument('--start_iter', type=int, default=0, help='the iteration to start training')
    parser.add_argument('--batch_size', type=int, default=8, help='mini batch number of images')
    parser.add_argument('--im_size', type=int, default=2048, help='image resolution')
    parser.add_argument('--ckpt', type=str, default='None', help='checkpoint weight path if have one')
    parser.add_argument('--workers', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('--save_interval', type=int, default=100, help='number of iterations to save model')
    parser.add_argument('--recalculate_porosity', action='store_true', default=False, help='force recalculation of porosity values even if file exists')

    args = parser.parse_args()
    print(args)

    train(args)