import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import resize
from scipy.interpolate import interp1d


import torch
from torchvision.datasets.folder import default_loader

from violet.utils.dataloaders import dino_he_transform


def transparent_cmap(cmap, n=255, a=.8):
    mycmap = cmap
    mycmap._init()
    mycmap._lut[:, -1] = np.linspace(0, a, n+4)

    return mycmap


def plot_attention(img, attn, num_patches=14, figsize=(10, 4), display='head',
                   cmap=plt.cm.Greens, alpha=.5):
    # img: (h, w, c)
    # attn: (n_heads, d, d)
    cm = transparent_cmap(cmap)

    # if displaying every head
    if display == 'head':
        fig, axs = plt.subplots(3, attn.shape[0], figsize=figsize)
        for i in range(attn.shape[0]):
            axs[1, i].imshow(img)

            # keep only patch attention
            head_attn = attn[i, 0, 1:]
            head_attn = head_attn.reshape(num_patches, num_patches)
            axs[0, i].imshow(img, alpha=alpha)
            axs[0, i].imshow(resize(head_attn,
                             (img.shape[0], img.shape[1])), cmap=cm)
            axs[2, i].imshow(head_attn, cmap=cm)
            for ax in axs[:, i]:
                ax.set_xticks([])
                ax.set_yticks([])

            axs[0, i].set_title(f'head {i}')

        axs[0, 0].set_ylabel('overlay')
        axs[1, 0].set_ylabel('image')
        axs[2, 0].set_ylabel('attention')
    # mean of all heads
    else:
        fig, axs = plt.subplots(3, 1, figsize=figsize)
        axs[1].imshow(img)
        mean_attn = attn[:, 0, 1:].reshape(attn.shape[0],
                                           num_patches, num_patches)
        mean_attn = mean_attn.mean(axis=0)
        axs[0].imshow(img, alpha=alpha)
        axs[0].imshow(resize(mean_attn, (img.shape[0], img.shape[1])), cmap=cm) 
        axs[2].imshow(mean_attn, cmap=cm)
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])

        axs[0].set_ylabel('overlay')
        axs[1].set_ylabel('image')
        axs[2].set_ylabel('attention')

    return fig, axs


def get_image_attention(img, model, apply_transform=True):
    transform = dino_he_transform()
    if isinstance(img, str):
        img = default_loader(img)

    if apply_transform:
        img = transform(img)

    # add batch dimension
    imgs = img.unsqueeze(0)

    # move to gpu if model is cuda
    if next(model.parameters()).is_cuda and not imgs.is_cuda:
        imgs = imgs.cuda()

    # make sure we are in eval mode
    model.eval()
    with torch.no_grad():
        attn = model.get_last_selfattention(imgs)

    # just 1 img
    attn = attn[0]

    # put to cpu if needed
    if attn.is_cuda:
        attn = attn.cpu()

    return attn.numpy()


def plot_image_attention(img, model, apply_transform=True, display='head',
                         alpha=.5):
    """
    Plot attention for img

    img can be either
        - filepath of image to plot attention for
        - uint8 array of shape (c, h, w)
    """
    attn = get_image_attention(img, model, apply_transform=True)

    # load img if needed
    if isinstance(img, str):
        img = np.asarray(default_loader(img))
    else:
        # make sure we are cpu/numpy
        if isinstance(img, torch.tensor):
            # move channel axis
            img = img.view(1, 2, 0)
            if img.is_cuda:
                img = img.cpu()
            img = img.numpy()

    interp = interp1d([np.min(img), np.max(img)], [0, 1])
    img = interp(img)

    return plot_attention(img, attn, display=display, alpha=alpha)
