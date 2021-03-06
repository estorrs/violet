from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def get_writer(log_dir):
    Path(log_dir).mkdir(exist_ok=True, parents=True)
    writer = SummaryWriter(log_dir)
    return writer


def collate_st_learner_params(img_dir, weights, samples, val_samples, gpu,
                              min_counts, frozen_lr, unfrozen_lr, run_dir,
                              resolution, train_dataloader, val_dataloader,
                              vit):
    return {
        'run_directory': run_dir,
        'dataset': {
            'image_directory': img_dir,
            'targets': list(train_dataloader.dataset.labels),
            'resolution': resolution,
            'min_counts': min_counts,
            'train_dataset': {
                'samples': [s for s in samples if s not in val_samples],
                'num_spots': len(train_dataloader.dataset.samples),
            },
            'val_dataset': {
                'samples': val_samples,
                'num_spots': len(val_dataloader.dataset.samples),
            },
        },
        'vit': {
            'pretrained_weights': weights,
            'patch_size': vit.vit.patch_embed.proj.kernel_size[0],
            'img_size': vit.vit.patch_embed.img_size,
            'total_patches': vit.vit.patch_embed.num_patches,
            'embed_dim': vit.vit.embed_dim,
        },
        'head': {
            'type': 'linear',
        },
        'hyperparams': {
            'batch_size': train_dataloader.batch_size,
            'loss': 'torch.nn.MSELoss',
            'frozen_lr': frozen_lr,
            'unfrozen_lr': unfrozen_lr,
            'scheduler': 'torch.optim.lr_scheduler.OneCycleLR',
            'opt': 'torch.optim.Adam',
            'gpu': gpu,
        },
    }


def collate_imagenet_st_learner_params(img_dir, weights, samples, val_samples,
                                       gpu, min_counts, frozen_lr, unfrozen_lr,
                                       run_dir, resolution, train_dataloader,
                                       val_dataloader, model, model_name):
    return {
        'run_directory': run_dir,
        'dataset': {
            'image_directory': img_dir,
            'targets': list(train_dataloader.dataset.labels),
            'resolution': resolution,
            'min_counts': min_counts,
            'train_dataset': {
                'samples': [s for s in samples if s not in val_samples],
                'num_spots': len(train_dataloader.dataset.samples),
            },
            'val_dataset': {
                'samples': val_samples,
                'num_spots': len(val_dataloader.dataset.samples),
            },
        },
        'convnet': {
            'imagenet_pretrained': True,
            'embed_dim': model.vit.fc.out_features,
            'model_name': model_name
        },
        'head': {
            'type': 'linear',
        },
        'hyperparams': {
            'batch_size': train_dataloader.batch_size,
            'loss': 'torch.nn.MSELoss',
            'frozen_lr': frozen_lr,
            'unfrozen_lr': unfrozen_lr,
            'scheduler': 'torch.optim.lr_scheduler.OneCycleLR',
            'opt': 'torch.optim.Adam',
            'gpu': gpu,
        },
    }
