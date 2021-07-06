import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from einops import rearrange


def create_pseudocolor_image(img, dataset, channels, colors=None):
    if isinstance(img, torch.Tensor):
        img = rearrange(img, 'c h w -> h w c').numpy()

    idxs = [dataset.channels.index(c) for c in channels]

    if colors is None:
        colors = sns.color_palette()

    new = np.zeros((img.shape[0], img.shape[1], 3))
    for i, c in enumerate(idxs):
        new += np.repeat(np.expand_dims(img[:, :, c], axis=-1),
                         3, axis=-1) * np.asarray(colors[i])

    new = (new / np.max(new)) * 255.
    return new.astype(np.uint8)
