import os
import re

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


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
