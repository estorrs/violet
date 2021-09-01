import os
import io
from pathlib import Path
import importlib.resources as pkg_resources

import pandas as pd
import numpy as np
import scanpy as sc
import tifffile
import staintools
from openslide import OpenSlide
from PIL import Image
from skimage.transform import rescale

import pkgutil

STAIN_REF = np.array(Image.open(io.BytesIO(
    pkgutil.get_data("violet", "data/stain_reference.jpeg")
)))


def normalize_counts(a, method='log'):
    if method == 'scanpy':
        sc.pp.normalize_total(a, inplace=True)
        sc.pp.log1p(a)
    elif method == 'log':
        sc.pp.log1p(a)
        if 'sparse' in str(type(a.X)):
            a.X = a.X.toarray()
        maxs = a.X.max(axis=0)
        maxs[maxs==0] = 1.
        a.X = a.X / maxs

    return a


def process_adata(sid, fp, markers, n_top_genes=200, min_counts=2500,
                  normalization='log'):
    a = sc.read_visium(fp)
    a.var_names_make_unique()
    a.var["mt"] = a.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(a, qc_vars=["mt"], inplace=True)
    sc.pp.filter_cells(a, min_counts=min_counts)
    a.obs.index = [f'{sid}_{x}' for x in a.obs.index]

    a = normalize_counts(a, method=normalization)

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


def normalize_he_image(img, ref=None, b=2000, coverage=.25):
    if ref is None:
        target = staintools.LuminosityStandardizer.standardize(STAIN_REF)
    else:
        target = staintools.LuminosityStandardizer.standardize(ref)
    normalizer = staintools.StainNormalizer(method='vahadane')
    normalizer.fit(target)

    nrows, ncols = (img.shape[0] // b) + 1, (img.shape[1] // b) + 1

    new = np.zeros(img.shape)

    for r in range(nrows):
        for c in range(ncols):
            im = img[r*b:(r+1)*b, c*b:(c+1)*b]
            if coverage is not None and not is_background(im, coverage=.75):
                to_transform = staintools.LuminosityStandardizer.standardize(im)
                x = normalizer.transform(to_transform)
                new[r*b:(r+1)*b, c*b:(c+1)*b] = x

    return new.astype(np.uint8)


def extract_st_tiles(data_map, normalize=True):
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

        if normalize:
            b = 2000 - (2000 % int(d))
            normalized_img = normalize_he_image(img, b=b)

        for (c, r), barcode in zip(sample_to_coords[s], sample_to_barcodes[s]):
            r1, r2 = int(r - (d * .5)),  int(r + (d * .5))
            c1, c2 = int(c - (d * .5)),  int(c + (d * .5))

            if normalize:
                imgs.append(normalized_img[r1:r2, c1:c2])
            else:
                imgs.append(img[r1:r2, c1:c2])
            img_ids.append(f'{s}_{barcode}')

    return imgs, img_ids


def get_svs_specs(svs_fp):
    o = OpenSlide(svs_fp)
    return o.dimensions, float(o.properties['aperio.MPP'])


def get_svs_tile_shape(svs_fp, resolution=55.):
    (r, c), mpp = get_svs_specs(svs_fp)
    tile_size = int(resolution // mpp)
    n_rows = r // tile_size
    n_cols = c // tile_size

    return (n_rows, n_cols), tile_size


def get_svs_array(svs_fp, scale=.05):
    o = OpenSlide(svs_fp)
    img = o.read_region((0, 0), 0, o.dimensions)
    img = img.resize((int(o.dimensions[0] * scale),
                      int(o.dimensions[1] * scale)))
    img = np.array(img)
    if img.shape[-1] > 3:
        img = img[:, :, :3]

    return img


def get_he_image(fp, scale=.05):
    if fp[-4:] == '.svs':
        img = get_svs_array(fp, scale=scale)
    else:
        img = tifffile.imread(fp)

        # convert to uint8
        img = img / np.max(img)
        img *= 255
        img = img.astype(np.uint8)

        img = rescale(img, scale, anti_aliasing=False)

    return img


def extract_svs_tiles(sample_to_svs, resolution=55., background_pct=.5,
                      normalize=True):
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

        if normalize:
            b = 2000 - (2000 % tile_size)
            normalized_img = normalize_he_image(img, b=b)

        for r in range(n_rows):
            for c in range(n_cols):
                r1, r2 = r * tile_size, (r + 1) * tile_size
                c1, c2 = c * tile_size, (c + 1) * tile_size
                tile = img[r1:r2, c1:c2]
                if tile.shape[0] == tile_size and tile.shape[1] == tile_size:
                    if not is_background(tile, coverage=background_pct):
                        if normalize and np.sum(normalized_img[r1:r2, c1:c2]) > 0.:
                            imgs.append(normalized_img[r1:r2, c1:c2])
                            img_ids.append(f'{sample}_{r}_{c}')
                        else:
                            imgs.append(tile)
                            img_ids.append(f'{sample}_{r}_{c}')

    return imgs, img_ids


def extract_tif_tiles(fp, resolution=55., mpp=1., normalize=True):
    img = tifffile.imread(fp)

    # convert to uint8
    img = img / np.max(img)
    img *= 255
    img = img.astype(np.uint8)


    tile_size = int(resolution // mpp)
    n_rows = img.shape[0] // tile_size
    n_cols = img.shape[1] // tile_size

    if normalize:
        b = (2000 - (2000 % tile_size)) % tile_size
        normalized_img = normalize_he_image(img, b=b)

    imgs, img_ids = [], []
    count = 0
    for r in range(n_rows):
        for c in range(n_cols):
            r1, r2 = r * tile_size, (r + 1) * tile_size
            c1, c2 = c * tile_size, (c + 1) * tile_size
            tile = img[r1:r2, c1:c2]

            if normalize and not np.count_nonzero(
                    np.where(np.sum(normalized_img[r1:r2, c1:c2], axis=-1) == 0.)):
                imgs.append(normalized_img[r1:r2, c1:c2])
                img_ids.append(f'{r}_{c}')
            else:
                imgs.append(tile)
                img_ids.append(f'{r}_{c}')
            count += 1

    return imgs, np.asarray(img_ids)


def extract_and_write_multichannel(
        sample_to_tifs, output_dir, val_samples=None, resolution=55., mpp=1.,
        val_split=.1, background_thres=100, background_channel='DAPI'):
    samples = sorted(sample_to_tifs.keys())
    channels = sorted(set.intersection(
        *[set(d.keys()) for d in sample_to_tifs.values()]))

    for sample in samples:
        for channel in channels:
            if val_samples is not None:
                if sample in val_samples:
                    Path(os.path.join(
                        output_dir, 'val', sample, channel)
                        ).mkdir(parents=True, exist_ok=True)
                else:
                    Path(os.path.join(
                        output_dir, 'train', sample, channel)
                        ).mkdir(parents=True, exist_ok=True)
            else:
                Path(os.path.join(
                    output_dir, 'train', sample, channel)
                    ).mkdir(parents=True, exist_ok=True)
                Path(os.path.join(
                    output_dir, 'val', sample, channel)
                    ).mkdir(parents=True, exist_ok=True)

    for sample in samples:
        # initialize training pool in case we need it
        train_img_ids = None

        if background_thres is not None:
            imgs, img_ids = extract_tif_tiles(sample_to_tifs[sample][background_channel],
                                              resolution=resolution, mpp=mpp)
            keep = {i for img, i in zip(imgs, img_ids)
                    if np.sum(img) > background_thres}
        for channel in channels:
            imgs, img_ids = extract_tif_tiles(sample_to_tifs[sample][channel],
                                              resolution=resolution, mpp=mpp)
            if background_thres is not None:
                imgs, img_ids = zip(*[(x, i) for x, i in zip(imgs, img_ids)
                                      if i in keep])
                imgs, img_ids = np.asarray(imgs), np.asarray(img_ids)

            if val_samples is None:
                if train_img_ids is None:
                    n = int(len(img_ids) * val_split)
                    pool = np.random.permutation(np.arange(len(img_ids)))
                    train_img_ids = set(img_ids[pool[:-n]])
                for img, img_id in zip(imgs, img_ids):
                    im = Image.fromarray(img)
                    if img_id in train_img_ids:
                        out = os.path.join(output_dir, 'train', sample, channel)
                    else:
                        out = os.path.join(output_dir, 'val', sample, channel)
                    im.save(os.path.join(out, f'{img_id}.tif'))
            else:
                if sample not in val_samples:
                    for img, img_id in zip(imgs, img_ids):
                        im = Image.fromarray(img)
                        out = os.path.join(output_dir, 'train', sample, channel)
                        im.save(os.path.join(out, f'{img_id}.tif'))
                else:
                    for img, img_id in zip(imgs, img_ids):
                        im = Image.fromarray(img)
                        out = os.path.join(output_dir, 'val', sample, channel)
                        im.save(os.path.join(out, f'{img_id}.tif'))
