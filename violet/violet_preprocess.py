import argparse
import logging
import os
from pathlib import Path

import numpy as np
from PIL import Image

from violet.utils.preprocessing import extract_st_tiles, extract_svs_tiles

parser = argparse.ArgumentParser()

parser.add_argument('input_type', default='svs', type=str,
                    choices=['svs', 'st'],
                    help='Directory location to write processed image tiles')
parser.add_argument('input_image', type=str,
                    help='Filepath of file to be preprocessed. \
Must be either a high resolution .tiff if input_type is st or a .svs file \
if input_type is svs.')
parser.add_argument('output_dir', type=str,
                    help='Directory location to write processed image tiles')

parser.add_argument('--spaceranger-outs', type=str,
                    help='Location of spaceranger outputs. Manditory if \
input_type is st.')
parser.add_argument('--reference', type=str,
                    choices=['kidney', 'pancreas', 'breast'],
                    help='Tissue type to use for H&E image normalization.')
parser.add_argument('--image-id', type=str,
                    help='Prefix to use when writing image tiles. If no \
image id is provided then it will be extracted from input file.')
parser.add_argument('--resolution', default=55, type=int,
                    help='Width of extracted tiles. Default is 55 microns.')


def check_inputs(args):
    ext = args.input_image.split('.')[-1]
    if args.input_type == 'st':
        valid_exts = ['tif', 'tiff']
        if ext not in valid_exts:
            raise RuntimeError(f'input_image must be one of the following extensions\
 for st input_type: {valid_exts}')
    if args.input_type == 'svs':
        valid_exts = ['svs']
        if ext not in valid_exts:
            raise RuntimeError(f'input_image must be one of the following extensions\
 for svs input_type: {valid_exts}')


def write_imgs(imgs, img_ids, out_dir, exclude_black=True):
    if exclude_black:
        imgs, imgs_ids = zip(
            *[(x, y) for x, y in zip(imgs, img_ids)
              if np.count_nonzero(np.sum(x, axis=-1) == 0) < 10])
        logging.info(f'{len(imgs)} tiles after filtering')

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for img, img_id in zip(imgs, img_ids):
        im = Image.fromarray(img)
        im.save(os.path.join(out_dir, f'{img_id}.jpeg'))


def main(args):
    check_inputs(args)
    logging.info(f'input type: {args.input_type}')

    if args.input_type == 'st':
        image_id = (args.image_id if args.image_id is not None
                    else args.input_image.split('/')[-1].split('.')[0])
        data_map = {image_id: {'tif': args.input_image,
                               'spatial': args.spaceranger_outs}}
        imgs, img_ids = extract_st_tiles(data_map, ref=args.reference)
        logging.info(f'{len(imgs)} tiles extracted')
        write_imgs(imgs, img_ids, args.output_dir)
    elif args.input_type == 'svs':
        image_id = (args.image_id if args.image_id is not None
                    else args.input_image.split('/')[-1].split('.')[0])
        data_map = {image_id: args.input_image}
        imgs, img_ids = extract_svs_tiles(
            data_map, resolution=float(args.resolution), ref=args.reference)
        write_imgs(imgs, img_ids, args.output_dir)
    logging.info('tiles written')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
