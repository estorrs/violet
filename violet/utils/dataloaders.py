import os
import re

import numpy as np

import torch
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader


def listfiles(folder, regex=None):
    """Return all files with the given regex in the given folder structure"""
    for root, folders, files in os.walk(folder):
        for filename in folders + files:
            if regex is None:
                yield os.path.join(root, filename)
            elif re.findall(regex, os.path.join(root, filename)):
                yield os.path.join(root, filename)

def dino_he_transform(resize=(224, 224)):
    return transforms.Compose([
        transforms.Resize(resize, interpolation=3),
        transforms.ToTensor(),
        # normalize by means and stds from ST histopathology dataset
        # rather than imagenet
        transforms.Normalize((0.6591, 0.5762, 0.7749), (0.2273, 0.2373, 0.1685))
    ])



class ImageRegressionDataset(torch.utils.data.Dataset):
    """
    Constructs image dataset where the targets are
    continous variables in target_df

    Assumes target_df index and images in root_dir
    have matching sample names
    """
    def __init__(self, root_dir, target_df, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        pool = set(target_df.index)
        self.imgs = sorted(listfiles(root_dir))
        self.imgs = [i for i in self.imgs
                     if i.split('/')[-1].split('.')[0] in pool]
        idxs = [i.split('/')[-1].split('.')[0] for i in self.imgs]
        target_df = target_df.loc[idxs]

        self.targets = target_df.values
        self.labels = np.asarray(target_df.columns)
        self.samples = np.asarray(target_df.index.to_list())

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_loc = self.imgs[idx]
        image = default_loader(img_loc)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def get_dataloader(img_dir, batch_size=64, drop_last=True, shuffle=True,
                   resize=(224, 224)):
    """
    Get basic image dataloader learning.

    Each subfolder in directory is treated as a class
    """

    transform = dino_he_transform(resize=resize)

    dataset = datasets.ImageFolder(img_dir, transform=transform)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        drop_last=drop_last,
    )

    return dataloader


def image_classification_dataloaders(root_dir, batch_size=64):
    """
    Get training and validation dataloaders for an image directory.

    The image directory should have subfolders 'train' and 'val'
    """
    train_dataloader = get_dataloader(
        os.path.join(root_dir, 'train'),
        batch_size=batch_size,
    )
    val_dataloader = get_dataloader(
        os.path.join(root_dir, 'val'),
        batch_size=batch_size,
        shuffle=False
    )

    return train_dataloader, val_dataloader


def image_regression_dataloaders(root_dir, target_df, transform=None,
                                 batch_size=64, resize=(224, 224)):
    """
    Get training and validation dataloaders for multivariate regression.

    The image directory should have subfolders 'train' and 'val'

    The target_df should contain regression targets
        - rows are samples. target_df index should be names of
        corresponding image in root_dir.
        - columns are target variables.
    """
    transform = dino_he_transform(resize=resize)

    train_dataset = ImageRegressionDataset(os.path.join(root_dir, 'train'),
                                           target_df, transform=transform)
    val_dataset = ImageRegressionDataset(os.path.join(root_dir, 'val'),
                                         target_df, transform=transform)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        drop_last=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=batch_size,
        drop_last=True,
    )

    return train_dataloader, val_dataloader
