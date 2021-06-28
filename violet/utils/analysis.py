import os
import re

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib import cm

from violet.utils.preprocessing import get_svs_tile_shape


def plot_image_umap(xs, ys, fps, figsize=(10, 10), zoom=.2):
    """Plots scatter plot with the given images"""
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(xs, ys)
    for x, y, fp in zip(xs, ys, fps):
        ab = AnnotationBbox(
            OffsetImage(plt.imread(fp), zoom=zoom),
            (x, y), frameon=False, )
        ax.add_artist(ab)

    ax.set_xticks([])
    ax.set_yticks([])

    return fig, ax


def retile_predictions(svs_fp, df, resolution=55.):
    """
    Retile predictions for an image.

    Prediction dataframe must be indexed as the following:
    <sample>_<row>_<col>

    Returns (h, w, c) array where c is the number of markers
    in the dataframe. c is ordered by the prediction dataframe column
    """
    to_sample = {(int(x.split('_')[-2]), int(x.split('_')[-1])): x
                 for x in df.index}

    (n_rows, n_cols), _ = get_svs_tile_shape(svs_fp, resolution=resolution)

    img = np.zeros((n_rows, n_cols, df.shape[1]), dtype=np.float32)
    for r in range(n_rows):
        for c in range(n_cols):
            if (r, c) in to_sample:
                img[r, c, :] = df.loc[to_sample[(r, c)]].to_numpy()

    return img


def display_predictions(he_img, df, tile_size, hue, scale, alpha=.5, s=1,
                        row_offset=0, col_offset=0, cmap=cm.Blues,
                        show_he=True):
    if show_he:
        plt.imshow(he_img)

    rs, cs, vals = [], [], []
    for i, row in df.iterrows():
        r, c = int(i.split('_')[-2]), int(i.split('_')[-1])
        rs.append(int(r * tile_size * scale) + int(row_offset * scale))
        cs.append(int(c * tile_size * scale) + int(col_offset * scale))
        vals.append(row[hue])

    plt.scatter(cs, rs, s=[s] * len(rs), c=vals, cmap=cmap, alpha=alpha)
