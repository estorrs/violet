import os

import pandas as pd
import numpy as np
import scanpy as sc
import tifffile
from openslide import OpenSlide


def process_adata(sid, fp, markers, n_top_genes=200, min_counts=2500):
    a = sc.read_visium(fp)
    a.var_names_make_unique()
    a.var["mt"] = a.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(a, qc_vars=["mt"], inplace=True)
    sc.pp.filter_cells(a, min_counts=min_counts)
    a.obs.index = [f'{sid}_{x}' for x in a.obs.index]
    sc.pp.normalize_total(a, inplace=True)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, flavor="seurat", n_top_genes=n_top_genes)

    f = a[:, markers]

    df = pd.DataFrame(f.X.toarray(), columns=f.var.index, index=f.obs.index)

    return df


def is_background(tile, thresh=220., coverage=.5):
    means = np.mean(tile, axis=-1)
    n_background = np.count_nonzero(means >= thresh)
    pct_background = n_background / (means.shape[0] * means.shape[1])
    if pct_background >= coverage:
        return True
    else:
        return False


def extract_st_tiles(data_map):
    """
    Extract H&E tiles from high resolution tif that corresponds to ST data

    Parameters
    ----------
    data_map: dict
        - dictionary mapping sample ids to their visium spaceranger outputs
        and high resolution tif file. Structure for each value is the
        following:
            {
                'spatial': <visium output fp>,
                'tif': <high res tif fp>
            }
    """
    sample_to_coords, sample_to_diameter = {}, {}
    sample_to_barcodes = {}
    for sample, d in data_map.items():
        a = sc.read_visium(d['spatial'])
        a.var_names_make_unique()
        coords = a.obsm['spatial']
        sample_to_coords[sample] = coords
        sample_to_barcodes[sample] = a.obs.index.to_list()

        sample_to_diameter[sample] = a.uns['spatial'][list(
            a.uns['spatial'].keys())[0]]['scalefactors']['spot_diameter_fullres']

    imgs, img_ids = [], []
    order = sorted(sample_to_coords.keys())
    for s in order:
        img = tifffile.imread(data_map[s]['tif'])
        d = sample_to_diameter[s]
        for (c, r), barcode in zip(sample_to_coords[s], sample_to_barcodes[s]):
            r1, r2 = int(r - (d * .5)),  int(r + (d * .5))
            c1, c2 = int(c - (d * .5)),  int(c + (d * .5))

            imgs.append(img[r1:r2, c1:c2])
            img_ids.append(f'{s}_{barcode}')

    return imgs, img_ids


def extract_svs_tiles(sample_to_svs, resolution=55.):
    imgs, img_ids = [], []
    for sample, svs in sample_to_svs.items():
        o = OpenSlide(svs)
        mpp = float(o.properties['aperio.MPP'])

        img = o.read_region((0, 0), 0, o.dimensions)
        img = np.array(img)
        if img.shape[-1] > 3:
            img = img[:, :, :3]

        tile_size = int(resolution // mpp)
        n_rows = img.shape[0] // tile_size
        n_cols = img.shape[1] // tile_size

        for r in range(n_rows):
            for c in range(n_cols):
                r1, r2 = r * tile_size, (r + 1) * tile_size
                c1, c2 = c * tile_size, (c + 1) * tile_size
                tile = img[r1:r2, c1:c2]
                if not is_background(tile):
                    imgs.append(tile)
                    img_ids.append(f'{sample}_{r}_{c}')

    return imgs, img_ids
