import json

import torch

from violet.models import vit_small
from violet.models.st import STRegressor
from violet.utils.dino_utils import load_pretrained_weights


def load_pretrained_model(pretrained_weights, model_name='vit_small',
                          patch_size=16):
    """
    Load pretrained DINO model.

    Note that this will load the teacher model, not the student.

    Currently only vit_small implemented
    """
    model = vit_small(num_classes=0)
    load_pretrained_weights(model, pretrained_weights, 'teacher', 'vit_small',
                            patch_size)

    return model


def load_trained_st_regressor(weights, summary):
    """
    Loads a trained regressor from a STLearner training run.

    Run must have saved checkpoint weights and a summary file.
    """
    vit = vit_small()

    regressor = STRegressor(vit, len(summary['dataset']['targets']))

    regressor.load_state_dict(torch.load(weights))

    return regressor


def predict(dataloader, model):
    """
    Utility function to collate predictions from a given
    model/dataloader
    """
    predictions = torch.zeros((len(dataloader.dataset.samples),
                               model.cls_token.shape[-1]))
    is_cuda = next(model.parameters()).is_cuda

    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            if is_cuda:
                x, y = x.cuda(), y.cuda()
            preds = model(x)
            predictions[i * dataloader.batch_size:(i+1) * dataloader.batch_size] = preds

    if predictions.is_cuda:
        predictions = predictions.cpu()

    return predictions.numpy()
