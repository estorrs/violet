import argparse
import os
import sys

from pathlib import Path

import torch
import torch.nn as nn

from vit_pytorch import ViT

import violet.utils as utils
from violet.models import Dino


def get_args_parser():
    parser = argparse.ArgumentParser('violet', add_help=False)
    parser.add_argument('--data-dir', type=str,
                        help="""Directory with train and validation data""")
    parser.add_argument('--out-dir', type=str,
                        help="""Output directory""")
    parser.add_argument('--seed', default=0,
                        type=int, help='Random seed.')
    parser.add_argument('--epochs', default=2,
                        type=int, help='Epochs to train for')

    return parser


def train_epoch(args, epoch, learner, opt, train_dataloader, val_dataloader,
                writer):
    train_loss, val_loss = 0., 0.

    learner.train()
    for i, (images, _) in enumerate(train_dataloader):
        if i % 10 == 0:
            print(epoch, i, len(train_dataloader))
        images = images.cuda()
        loss = learner(images)
        opt.zero_grad()
        loss.backward()
        opt.step()
        learner.update_moving_average()  # update moving average of teacher encoder and teacher centers

        train_loss += loss

    learner.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(val_dataloader):
            images = images.cuda()
            loss = learner(images)
            learner.update_moving_average()  # update moving average of teacher encoder and teacher centers

            val_loss += loss

    train_loss = train_loss / len(train_dataloader)
    val_loss = val_loss / len(val_dataloader)

    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)
    print(f'epoch: {epoch}, train loss: {train_loss}, val loss: {val_loss}')


def train_dino(args):
    """Train dino"""
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)

    log_dir = os.path.join(args.out_dir, 'logs')
    writer = utils.get_writer(log_dir)

    model = ViT(
        image_size=256,
        patch_size=16,
        num_classes=1000,
        dim=1024,
        depth=6,
        heads=8,
        mlp_dim=2048
    )
    model = model.cuda()
##     model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    learner = Dino(
        model,
        image_size=256,
        hidden_layer='to_latent',         # hidden layer name or index, from which to extract the embedding
        projection_hidden_size=256,       # projector network hidden dimension
        projection_layers=4,              # number of layers in projection network
        num_classes_K=65336,              # output logits dimensions (referenced as K in paper)
        student_temp=0.9,                 # student temperature
        teacher_temp=0.04,                # teacher temperature, needs to be annealed from 0.04 to 0.07 over 30 epochs
        local_upper_crop_scale=0.4,       # upper bound for local crop - 0.4 was recommended in the paper 
        global_lower_crop_scale=0.5,      # lower bound for global crop - 0.5 was recommended in the paper
        moving_average_decay=0.9,         # moving average of encoder - paper showed anywhere from 0.9 to 0.999 was ok
        center_moving_average_decay=0.9,  # moving average of teacher centers - paper showed anywhere from 0.9 to 0.999 was ok
    )
    learner = learner.cuda()
##     learner = nn.parallel.DistributedDataParallel(
##         learner, device_ids=[args.gpu]
##     )

    train_dataloader, val_dataloader = utils.get_dataloaders(
        args.data_dir, batch_size=64, distributed=True
    )

    opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

    for epoch in range(args.epochs):
        train_epoch(args, epoch, learner, opt, train_dataloader,
                    val_dataloader, writer)

        print('saving checkpoint at',
              os.path.join(args.out_dir, f'checkpoint_{epoch}.pt'))
        torch.save(model.state_dict(),
                   os.path.join(args.out_dir, f'checkpoint_{epoch}.pt'))

    writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('violet', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    train_dino(args)
