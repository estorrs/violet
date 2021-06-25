import pprint
import torch
import pandas as pd

from violet.models.st import STRegressor, STLearner
from violet.utils.model import load_pretrained_model
from violet.utils.dataloaders import image_regression_dataloaders
from violet.utils.logging import collate_st_learner_params
from violet.utils.preprocessing import process_adata


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
                    max_lr=1e-4):
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
        min_counts, max_lr, run_dir, train_dataloader, val_dataloader,
        regressor)
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
