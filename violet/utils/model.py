import torch

from violet.models import vit_small
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


def predict(dataloader, model):
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
