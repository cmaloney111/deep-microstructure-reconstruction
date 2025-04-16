import os
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import Dataset
from PIL import Image
from copy import deepcopy
import shutil
import json

def InfiniteSampler(n):
    """Data sampler"""
    # check if the number of samples is valid
    if n <= 0:
        raise ValueError(f"Invalid number of samples: {n}.\nMake sure that images are present in the given path.")
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


class InfiniteSamplerWrapper(data.sampler.Sampler):
    """Data sampler wrapper"""
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31


def copy_G_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten
    

def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def get_dir(args):
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        
    task_name = os.path.join(args.output_path, 'train_results', args.name)
    saved_model_folder = os.path.join(task_name, 'models')
    saved_image_folder = os.path.join(task_name, 'images')
    
    os.makedirs(saved_model_folder, exist_ok=True)
    os.makedirs(saved_image_folder, exist_ok=True)

    for f in os.listdir('./'):
        if '.py' in f:
            shutil.copy(f, os.path.join(task_name, f))
    
    with open(os.path.join(saved_model_folder, '../args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    return saved_model_folder, saved_image_folder


class ImageFolder(Dataset):
    """Simple dataset for loading images from a folder"""
    def __init__(self, root, transform=None):
        super(ImageFolder, self).__init__()
        self.root = root
        self.frame = self._parse_frame()
        self.transform = transform
        self.class_map = None  # No class mapping for basic ImageFolder

    def _parse_frame(self):
        frame = []
        img_names = os.listdir(self.root)
        img_names.sort()
        for i in range(len(img_names)):
            image_path = os.path.join(self.root, img_names[i])
            if image_path[-4:] == '.jpg' or image_path[-4:] == '.png' or image_path[-4:] == '.tif' or image_path[-5:] == '.jpeg': 
                frame.append(image_path)
        return frame

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        file = self.frame[idx]
        img = Image.open(file).convert('RGB')
            
        if self.transform:
            img = self.transform(img) 

        return img


class ClassConditionedDataset(Dataset):
    """Dataset for class conditioned GAN - loads images from subdirectories where each subdirectory is a class"""
    def __init__(self, root, transform=None):
        super(ClassConditionedDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.classes, self.class_to_idx = self._find_classes()
        self.frame, self.targets = self._parse_frame()
        
    def _find_classes(self):
        """
        Find class folder names in root directory
        """
        classes = [d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
    
    def _parse_frame(self):
        """
        Create list of file paths and corresponding class indices
        """
        frame = []
        targets = []
        
        for class_name in self.classes:

            class_idx = self.class_to_idx[class_name]
            class_dir = os.path.join(self.root, class_name)
            
            # Get all valid images in this class directory
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if img_path[-4:] == '.jpg' or img_path[-4:] == '.png' or img_path[-4:] == '.tif' or img_path[-5:] == '.jpeg':
                    frame.append(img_path)
                    targets.append(class_idx)
                    
        return frame, targets

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        file = self.frame[idx]
        target = self.targets[idx]
        
        img = Image.open(file).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, target


def calculate_image_porosity(img_tensor):
    """
    Calculate porosity from an image tensor
    
    Args:
        img_tensor: Tensor of shape [C, H, W] with values in [-1, 1]
        
    Returns:
        Porosity value (scalar) between 0 and 1
    """
    # Convert to [0, 1] range
    img = (img_tensor + 1) / 2
    
    # Convert to grayscale if RGB
    if img.size(0) == 3:
        # Use luminance formula
        img_gray = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
    else:
        img_gray = img.squeeze(0)
    
    # Apply threshold to create binary image (pores are assumed to be dark)
    # Adjust threshold if needed for your specific microstructure images
    threshold = 0.5
    binary = (img_gray < threshold).float()
    
    # Calculate porosity as ratio of pore pixels to total pixels
    porosity = binary.sum() / (binary.size(0) * binary.size(1))
    
    return porosity


class BatchPorositySampler(data.sampler.Sampler):
    """
    Sampler that tries to create batches with similar porosity
    
    This helps in training the conditional GAN to respect porosity characteristics
    """
    def __init__(self, dataset, batch_size, porosity_tolerance=0.05):
        self.dataset = dataset
        self.batch_size = batch_size
        self.tolerance = porosity_tolerance
        self.porosities = self._calculate_porosities()
        self.indices_by_porosity = self._group_by_porosity()
        
    def _calculate_porosities(self):
        """Calculate porosity for all images in the dataset"""
        porosities = []
        for i in range(len(self.dataset)):
            if isinstance(self.dataset[i], tuple):
                img = self.dataset[i][0]
            else:
                img = self.dataset[i]
            porosity = calculate_image_porosity(img)
            porosities.append(porosity.item())
        return porosities
    
    def _group_by_porosity(self):
        """Group indices by similar porosity values"""
        indices_by_porosity = {}
        
        # Round porosities to group similar values
        for i, porosity in enumerate(self.porosities):
            rounded = round(porosity / self.tolerance) * self.tolerance
            if rounded not in indices_by_porosity:
                indices_by_porosity[rounded] = []
            indices_by_porosity[rounded].append(i)
            
        return indices_by_porosity
    
    def __iter__(self):
        # Create batches with similar porosity
        batches = []
        
        # Get list of porosity groups
        porosity_groups = list(self.indices_by_porosity.keys())
        
        # Shuffle the order of porosity groups
        np.random.shuffle(porosity_groups)
        
        for porosity in porosity_groups:
            indices = self.indices_by_porosity[porosity]
            # Shuffle indices within each porosity group
            np.random.shuffle(indices)
            
            # Create batches from this porosity group
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                # If we don't have enough samples to form a complete batch
                if len(batch_indices) < self.batch_size:
                    # Fill remaining spots with random samples from other groups
                    remaining = self.batch_size - len(batch_indices)
                    all_other_indices = []
                    for p in porosity_groups:
                        if p != porosity:
                            all_other_indices.extend(self.indices_by_porosity[p])
                    
                    if len(all_other_indices) >= remaining:
                        np.random.shuffle(all_other_indices)
                        batch_indices.extend(all_other_indices[:remaining])
                    else:
                        # Not enough samples, just duplicate some
                        while len(batch_indices) < self.batch_size:
                            batch_indices.append(np.random.choice(batch_indices))
                
                batches.append(batch_indices)
        
        # Shuffle the order of batches
        np.random.shuffle(batches)
        
        # Flatten batches to yield indices
        for batch in batches:
            for idx in batch:
                yield idx
    
    def __len__(self):
        return len(self.dataset)