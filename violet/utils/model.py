import torch

from violet.models.vit import vit_small
from violet.utils.dino_utils import load_pretrained_weights


def load_pretrained_model(pretrained_weights, model_name='vit_small',
                          patch_size=16, in_chans=3):
    """
    Load pretrained DINO model.

    Note that this will load the teacher model, not the student.

    Currently only vit_small implemented
    """
    model = vit_small(num_classes=0, in_chans=in_chans)
    load_pretrained_weights(model, pretrained_weights, 'teacher', 'vit_small',
                            patch_size)

    return model


def predict(dataloader, model, out_dim=None):
    """
    Utility function to collate predictions from a given
    model/dataloader.

    If out_dim not provided then assumes model is a vit
    that has output dim of size cls_token
    """
    if out_dim is None:
        out_dim = model.cls_token.shape[-1]

    predictions = torch.zeros((len(dataloader.dataset.samples), out_dim))
    is_cuda = next(model.parameters()).is_cuda

    model.eval()
    with torch.no_grad():
        # if there are targets in dataloader
        # refactor eventually so wont break for batch size of 2
        if len(next(iter(dataloader))) == 2:
            for i, (x, y) in enumerate(dataloader):
                if is_cuda:
                    x, y = x.cuda(), y.cuda()
                preds = model(x)
                r1 = i * dataloader.batch_size
                r2 = (i+1) * dataloader.batch_size
                predictions[r1:r2] = preds
        # otherwise assume single
        else:
            for i, x in enumerate(dataloader):
                if is_cuda:
                    x = x.cuda()
                preds = model(x)
                r1 = i * dataloader.batch_size
                r2 = (i+1) * dataloader.batch_size
                predictions[r1:r2] = preds

    if predictions.is_cuda:
        predictions = predictions.cpu()

    return predictions.numpy()
