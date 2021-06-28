import json
import os

import pprint
import torch
import pandas as pd
import scanpy as sc
from PIL import Image

from violet.models import vit_small
from violet.models.st import STRegressor, STLearner
from violet.utils.model import load_pretrained_model, predict
from violet.utils.dataloaders import (
    prediction_dataloader, image_regression_dataloaders)
from violet.utils.logging import collate_st_learner_params
from violet.utils.preprocessing import (
    process_adata, extract_st_tiles, extract_svs_tiles)


def get_target_df(adata_map, markers, min_counts=2500, n_top_genes=200):
    targets = None
    for s, fp in adata_map.items():
        df = process_adata(s, fp, markers, n_top_genes=n_top_genes,
                           min_counts=min_counts)

        if targets is None:
            targets = df
        else:
            targets = pd.concat((targets, df), axis=0)

    return targets


def load_st_learner(img_dir, weights, adata_map, run_dir, val_samples=None,
                    gpu=True, targets=None, min_counts=2500,
                    max_lr=1e-4, resolution=55.):
    """
    Utility function for loading a STLearner whose base is a vit
    with the associated weights.

    The STlearner can then be used to train a finetuned model for
    ST expression prediction from HE tiles.

    Parameters
    ----------
    img_dir: str
        - Directory containing HE image tiles. Image filenames
        (not including the file extension) must correspond with the sample name
        in adata_map. Image filenames have the following format:
        <sample_id>_<barcode>.<file extension>
    weights: str
        - Filepath to weights of pretrained vit
    adata_map: dict
        - Keys are sample_id. Values are filepath of visium spaceranger
        outs. I.e. the directory read by sc.read_visium().
    run_dir: str
        - ry files will be produced. Directory path that checkpoints,
        summary files, and tensorboardX output will be written to.
    val_samples: list
        - Samples ids to be used as validation dataset. All others will be
        used in training. If None, will take an 80/20 train/val split
        across all data.
    gpu: bool
        - Whether to use gpu or not. If system does not have a gpu
        this should be set to False.
    targets: list
        - Genes to use as target variables.
    min_counts: int
        - Spots with < min_counts will be excluded from the dataset
    max_lr: float
        - Max learning rate for cos scheduler.
    """
    target_df = get_target_df(adata_map, targets, min_counts=min_counts)

    val_regexs = [r'.*' + s for s in val_samples]
    train_dataloader, val_dataloader = image_regression_dataloaders(
            img_dir, target_df, val_regexs=val_regexs)

    model = load_pretrained_model(weights)
    regressor = STRegressor(model, len(train_dataloader.dataset.labels))
    if gpu:
        regressor = regressor.cuda()

    summary = collate_st_learner_params(
        img_dir, weights, sorted(adata_map.keys()), val_samples, gpu,
        min_counts, max_lr, run_dir, resolution,
        train_dataloader, val_dataloader, regressor)
    print('ST Learner summary:')
    pprint.pprint(summary)
    learner = STLearner(regressor, train_dataloader, val_dataloader,
                        run_dir, max_lr=max_lr, summary=summary)

    return learner


def run_st_learner(learner, frozen_epochs, unfrozen_epochs):
    print(f'Training frozen vit for {frozen_epochs} epochs')
    learner.fit(frozen_epochs)

    chkpt_fp = learner.save_checkpoint()
    print(f'Saved checkpoint at {chkpt_fp}')

    print('Unfreezing weights')
    learner.unfreeze_vit()

    print(f'Training unfrozen vit for {unfrozen_epochs} epochs')
    learner.fit(unfrozen_epochs)

    chkpt_fp, summary_fp = learner.save_final()
    print(f'Saved final checkpoint at {chkpt_fp}')
    print(f'Saved summary at {summary_fp}')


def load_trained_st_regressor(weights, summary):
    """
    Loads a trained regressor from a STLearner training run.

    Run must have saved checkpoint weights and a summary file.
    """
    vit = vit_small()

    regressor = STRegressor(vit, len(summary['dataset']['targets']))

    regressor.load_state_dict(torch.load(weights))

    return regressor


def predict_he_tiles(img_dir, weights, summary, regexs=None, batch_size=64,
                     gpu=True):
    """
    Predict HE tiles in for given filepaths.
    """
    if regexs is None:
        regexs = []

    # add file extensions
    regexs += [r'.*jpeg$', r'.*jpg$', r'.*png$', r'.*tif$', r'.*tiff$']

    dataloader = prediction_dataloader(img_dir, batch_size=batch_size,
                                       include_regexs=regexs, pad=True)

    meta = json.load(open(summary))
    regressor = load_trained_st_regressor(weights, meta)
    if gpu:
        regressor = regressor.cuda()

    preds = predict(dataloader, regressor,
                    out_dim=len(meta['dataset']['targets']))

    preds = pd.DataFrame(data=preds, columns=meta['dataset']['targets'],
                         index=dataloader.dataset.samples)

    # remove extra padding
    preds = preds[[True if x[:5] != '<pad_' else False
                   for x in preds.index]]

    return preds


def predict_visium(spatial_fp, high_res_fp, weights, summary,
                   tmp_dir=os.getcwd(), gpu=True):
    """
    Utility function for predicting visium spaceranger output objects.

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
    tmp_dir: str
        - Directory to write temporary files during run. Defaults to
        current working directory.
    """
    # not the most memory efficient
    # need to refactor
    data_map = {
        'inference_sample': {
            'spatial': spatial_fp, 'tif': high_res_fp
        }
    }
    imgs, img_ids = extract_st_tiles(data_map)

    fps = []
    for img, img_id in zip(imgs, img_ids):
        im = Image.fromarray(img)
        fp = os.path.join(tmp_dir, f'{img_id}.jpeg')
        im.save(fp)
        fps.append(fp)

    preds = predict_he_tiles(fps, weights, summary, gpu=gpu)

    # remove tmp files
    for fp in fps:
        os.remove(fp)

    # prep anndata object
    adata = sc.read_visium(spatial_fp)
    adata.var_names_make_unique()
    # switch back to only barcodes so index matchs anndata
    preds.index = [x.split('_')[-1] for x in preds.index]
    # rename cols
    preds.columns = [f'predicted_{c}' for c in preds.columns]
    adata.obs = pd.merge(adata.obs, preds.loc[adata.obs.index],
                         left_index=True, right_index=True)

    return adata


def predict_svs(svs_fp, weights, summary, tmp_dir=os.getcwd(),
                gpu=True, res=55., background_pct=.5):
    data_map = {svs_fp.split('/')[-1].split('.')[0]: svs_fp}
    sum = json.load(open(summary))
    r = sum['dataset']['resolution'] if 'resolution' in sum['dataset'] else res
    imgs, img_ids = extract_svs_tiles(data_map, resolution=r,
                                      background_pct=background_pct)
    print(f'predicting {len(imgs)} tiles.')

    fps = []
    for img, img_id in zip(imgs, img_ids):
        im = Image.fromarray(img)
        fp = os.path.join(tmp_dir, f'{img_id}.jpeg')
        im.save(fp)
        fps.append(fp)

    preds = predict_he_tiles(fps, weights, summary, gpu=gpu)

    # remove tmp files
    for fp in fps:
        os.remove(fp)

    return preds
