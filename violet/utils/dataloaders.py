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
        transforms.Normalize(
            (0.6591, 0.5762, 0.7749), (0.2273, 0.2373, 0.1685))
    ])


def imagenet_he_transform(resize=(224, 224)):
    return transforms.Compose([
        transforms.Resize(resize, interpolation=3),
        transforms.ToTensor(),
        # normalize by means and stds from imagenet since we are
        # using imagenet pretrained model
        transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


class ImagePredictionDataset(torch.utils.data.Dataset):
    """
    Constructs image dataset from image directory if root_dir is a
    directory. root_dir can also be a list of image filepaths

    If include or exclude regexs is a list only filenames with those patterns
    will be used.

    If pad is true then pad the last batch so it has a length of batch_size
    """
    def __init__(self, root_dir, transform=None, pad=True,
                 exclude_regexs=None, include_regexs=None,
                 batch_size=64):
        self.root_dir = root_dir
        self.transform = transform

        if isinstance(root_dir, str):
            self.imgs = sorted(listfiles(root_dir))
        else:
            self.imgs = sorted(self.root_dir)

        if include_regexs is not None:
            self.imgs = [i for i in self.imgs
                         if any([re.match(x, i) is not None
                                 for x in include_regexs])]

        if exclude_regexs is not None:
            self.imgs = [i for i in self.imgs
                         if not any([re.match(x, i) is not None
                                     for x in exclude_regexs])]

        idxs = [i.split('/')[-1].split('.')[0] for i in self.imgs]

        if pad:
            n = batch_size - (len(idxs) % batch_size)
            for i in range(n):
                idxs.append(f'<pad_{i}>')
                self.imgs.append(self.imgs[0])

        self.samples = np.asarray(idxs)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_loc = self.imgs[idx]
        image = default_loader(img_loc)
        if self.transform is not None:
            image = self.transform(image)
        return image


class ImageRegressionDataset(torch.utils.data.Dataset):
    """
    Constructs image dataset where the targets are
    continous variables in target_df

    Assumes target_df index and images in root_dir
    have matching sample names

    If regexs is a list only filenames with those patterns
    will be used
    """
    def __init__(self, root_dir, target_df, transform=None,
                 exclude_regexs=None, include_regexs=None):
        self.root_dir = root_dir
        self.transform = transform

        pool = set(target_df.index)
        self.imgs = sorted(listfiles(root_dir))

        self.imgs = [i for i in self.imgs
                     if i.split('/')[-1].split('.')[0] in pool]

        if include_regexs is not None:
            self.imgs = [i for i in self.imgs
                         if any([re.match(x, i) is not None
                                 for x in include_regexs])]

        if exclude_regexs is not None:
            self.imgs = [i for i in self.imgs
                         if not any([re.match(x, i) is not None
                                     for x in exclude_regexs])]

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
                                 batch_size=64, resize=(224, 224),
                                 val_regexs=None):
    """
    Get training and validation dataloaders for multivariate regression.

    The image directory should have subfolders 'train' and 'val'
    Or optionally with val_regexs you can include regexs for image names you
    would like in the validation set, all other samples will go in the training
    dataset.

    The target_df should contain regression targets. The images in root
    directory will be automatically filtered to match those specified in
    target_df.
        - rows are samples. target_df index should be names of
        corresponding image in root_dir, but with file extension removed.
        - columns are target variables.
    """
    if transform is None:
        transform = dino_he_transform(resize=resize)

    if val_regexs is None:
        train_dataset = ImageRegressionDataset(os.path.join(root_dir, 'train'),
                                               target_df, transform=transform)
        val_dataset = ImageRegressionDataset(os.path.join(root_dir, 'val'),
                                             target_df, transform=transform)
    else:
        train_dataset = ImageRegressionDataset(root_dir,
                                               target_df, transform=transform,
                                               exclude_regexs=val_regexs)
        val_dataset = ImageRegressionDataset(root_dir,
                                             target_df, transform=transform,
                                             include_regexs=val_regexs)

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


def prediction_dataloader(root_dir, transform=None,
                          batch_size=64, resize=(224, 224), pad=True,
                          include_regexs=None, exclude_regexs=None):
    """
    Get dataloader for image prediction

    If include or exclude regexs is a list those regexs will be used to
    filter the filepaths in root_dir

    if pad is true will add extra images to last batch so it is not dropped
    """
    if transform is None:
        transform = dino_he_transform(resize=resize)

    dataset = ImagePredictionDataset(
            root_dir, transform=transform, pad=True, batch_size=batch_size,
            exclude_regexs=exclude_regexs, include_regexs=include_regexs)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size,
        drop_last=True,
    )

    return dataloader
