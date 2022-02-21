import os
import re

import numpy as np
import torch
from torchvision import datasets, transforms
import torchvision.transforms.functional as F
from torchvision.datasets.folder import default_loader
from stainaug import Augmentor

from violet.utils.dino_utils import DataAugmentationDINOMultichannel, HE_color_transform


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
            (0.76806694, 0.47375619, 0.58864233), (0.17746654, 0.21851493, 0.18837758)
        )
    ])


def dino_he_transform_with_aug(resize=(224, 224)):
    return transforms.Compose([
        transforms.Resize(resize, interpolation=3),
        transforms.ToTensor(),
        HE_color_transform(),
        # normalize by means and stds from ST histopathology dataset
        # rather than imagenet
        transforms.Normalize(
            (0.74153281, 0.42759532, 0.67904226), (0.15070522, 0.16646285, 0.14752357)
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
##         transforms.RandomApply(
##             [transforms.ColorJitter(brightness=0.4, contrast=0.4,
##                                     saturation=0.2, hue=0.1)],
##             p=0.8
##         ),
##         transforms.RandomApply(
##             [transforms.ColorJitter(brightness=0.4, contrast=0.0,
##                                     saturation=0.0, hue=0.1)],
##             p=0.8
##         ),
    ])


def dino_multichannel_transform(resize=(224, 224)):
    return transforms.Compose([
        transforms.Resize(resize, interpolation=3),
        transforms.Grayscale(),
        transforms.ToTensor(),
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


def image_classification_dataloaders(root_dir, batch_size=64,
                                     resize=(224, 224)):
    """
    Get training and validation dataloaders for an image directory.

    The image directory should have subfolders 'train' and 'val'
    """
    train_dataloader = get_dataloader(
        os.path.join(root_dir, 'train'),
        batch_size=batch_size,
        transform=dino_he_transform_with_aug(resize=resize),
    )
    val_dataloader = get_dataloader(
        os.path.join(root_dir, 'val'),
        batch_size=batch_size,
        shuffle=False,
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
        train_transform = dino_he_transform_with_aug(resize=resize)
        val_transform = dino_he_transform(resize=resize)

    if val_regexs is None:
        train_dataset = ImageRegressionDataset(os.path.join(root_dir, 'train'),
                                               target_df, transform=train_transform)
        val_dataset = ImageRegressionDataset(os.path.join(root_dir, 'val'),
                                             target_df, transform=val_transform)
    else:
        train_dataset = ImageRegressionDataset(root_dir,
                                               target_df, transform=train_transform,
                                               exclude_regexs=val_regexs)
        val_dataset = ImageRegressionDataset(root_dir,
                                             target_df, transform=val_transform,
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


# Multichannel dataloaders

class MultichannelImageDataset(torch.utils.data.Dataset):
    """
    If include or exclude regexs is a list only filenames with those patterns
    will be used.

    If pad is true then pad the last batch so it has a length of batch_size
    """
    def __init__(self, root_dir, transform=None, pad=True,
                 exclude_regexs=None, include_regexs=None,
                 batch_size=64, return_dummy_y=False):
        self.root_dir = root_dir
        self.transform = transform
        self.load_image_transform = dino_multichannel_transform()
        self.return_dummy_y = return_dummy_y

        if isinstance(root_dir, str):
            self.imgs = sorted(listfiles(root_dir, regex='.tif$'))
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

        self.channels = set()
        sample_to_pos_to_channel = {}
        for i in self.imgs:
            sample = i.split('/')[-3]
            channel = i.split('/')[-2]
            position = i.split('/')[-1].split('.')[0]
            if sample not in sample_to_pos_to_channel:
                sample_to_pos_to_channel[sample] = {}
            if position not in sample_to_pos_to_channel[sample]:
                sample_to_pos_to_channel[sample][position] = {}

            sample_to_pos_to_channel[sample][position][channel] = i
            self.channels.add(channel)
        self.channels = sorted(self.channels)

        self.imgs = []
        self.samples = []
        for sample, d in sample_to_pos_to_channel.items():
            for pos, c_dict in d.items():
                self.imgs.append([c_dict[c] for c in self.channels])
                self.samples.append(f'{sample}_{pos}')

        if pad:
            n = batch_size - (len(self.samples) % batch_size)
            for i in range(n):
                self.samples.append(f'<pad_{i}>')
                self.imgs.append(self.imgs[0])

        self.imgs = np.asarray(self.imgs)
        self.samples = np.asarray(self.samples)

    def _load_image(self, fps):
        imgs = [default_loader(fp) for fp in fps]
        block = torch.cat([self.load_image_transform(img) for img in imgs],
                          dim=0)
        if self.transform is not None:
            block = self.transform(block)
        return block

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_locs = self.imgs[idx]
        block = self._load_image(img_locs)

        if self.return_dummy_y:
            return block, torch.zeros((1, 1))

        return block


def multichannel_image_dataloader(root_dir, batch_size=64, transform=None,
                                       resize=(224, 224), pad=True,
                                       shuffle=False):
    """
    Get training and validation dataloaders for an image directory.
    """
    dataset = MultichannelImageDataset(root_dir, transform=transform, pad=False)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size,
        drop_last=not pad,
    )

    return dataloader
