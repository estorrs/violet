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


def retile_multichannel_image(labels, pseudos):
    rs, cs = zip(*[(int(l.split('_')[-2]), int(l.split('_')[-1]))
                   for l in labels])

    tile_height, tile_width, _ = pseudos[0].shape
    retiled = np.zeros(((np.max(rs) + 1) * tile_height, (np.max(cs) + 1) * tile_width, 3))
    for r, c, img in zip(rs, cs, pseudos):
        r1, r2 = r * tile_height, (r + 1) * tile_height
        c1, c2 = c * tile_width, (c + 1) * tile_width
        retiled[r1:r2, c1:c2, :] = img

    return retiled.astype(np.uint8)
