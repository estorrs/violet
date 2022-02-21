import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn

from violet.utils.logging import get_writer


class STRegressor(nn.Module):
    """
    Regressor for finetuning ST model.

    Starts from model, which is DINO teacher.

    Attaches a single linear layer as classification head.
    """
    def __init__(self, model, num_classes, out_features=None):
        super().__init__()

        # doesn't necessary need to be ViT. Although if it is not a ViT
        # out_features will need to be passed in to specify in dimension
        # for head
        self.vit = model
        n = self.vit.num_features if out_features is None else out_features
        self.head = nn.Linear(n, num_classes)

        # freeze vit weights
        for param in self.vit.parameters():
            param.requires_grad = False

    def unfreeze_vit(self):
        for param in self.vit.parameters():
            param.requires_grad = True

    def forward(self, x):
        cls_token = self.vit(x)
        return self.head(cls_token)


class STLearner(object):
    """
    Learner for STRegressor
    """
    def __init__(self, model, train_dataloader, val_dataloader, run_dir,
                 frozen_lr=1e-4, unfrozen_lr=1e-4, verbose=True, summary=None):
        self.model = model
        self.frozen_lr = frozen_lr
        self.unfrozen_lr = unfrozen_lr
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.phase = '1.finetune_frozen_vit'
        self.verbose = verbose

        # logging and model outputs
        self.run_dir = run_dir
        self.checkpoint_dir = os.path.join(run_dir, 'checkpoints')
        self.log_dir = os.path.join(run_dir, 'logs')
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        self.writer = get_writer(self.log_dir)

        if summary is None:
            self.summary = {}
        else:
            self.summary = summary

        self.loss = nn.MSELoss()

        self.is_cuda = next(self.model.parameters()).is_cuda

        self.start = time.time()

    def _get_optimizer(self, epochs):
        if self.phase == '1.finetune_frozen_vit':
            lr = self.frozen_lr
        else:
            lr = self.unfrozen_lr

        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=lr,
            steps_per_epoch=len(self.train_dataloader), epochs=epochs)

        return opt, scheduler

    def _summarize(self, epoch, train_loss, val_loss, time_delta):
        if 'training' not in self.summary:
            self.summary['training'] = {}
        if self.phase not in self.summary['training']:
            self.summary['training'][self.phase] = {}
        self.summary['training'][self.phase][epoch] = {
            'train_loss': train_loss.tolist(),
            'val_loss': val_loss.tolist(),
            'time_elapsed': time_delta
        }

    def unfreeze_vit(self):
        self.model.unfreeze_vit()
        self.phase = '2.finetune_unfrozen_vit'

    def save_checkpoint(self, tag=''):
        chkpt = f'{self.phase}_{tag}.pth'
        save_path = os.path.join(self.checkpoint_dir, chkpt)
        print(f'saving checkpoint at {save_path}')
        torch.save(self.model.state_dict(), save_path)

        return save_path

    def save_final(self):
        save_path = os.path.join(self.checkpoint_dir, 'final.pth')
        torch.save(self.model.state_dict(), save_path)

        # add train time
        self.summary['training']['time_elapsed'] = time.time() - self.start
        summary_path = os.path.join(self.run_dir, 'summary.json')
        json.dump(self.summary, open(summary_path, 'w'))

        return save_path, summary_path

    def fit(self, epochs, save_every=None):
        opt, scheduler = self._get_optimizer(epochs)

        for epoch in range(epochs):
            train_loss, val_loss = 0., 0.
            start = time.time()

            self.model.train()
            for i, (x, y) in enumerate(self.train_dataloader):
                if self.is_cuda:
                    x, y = x.cuda(), y.cuda()
                logits = self.model(x)
                loss = self.loss(logits, y)
                opt.zero_grad()
                loss.backward()
                opt.step()

                train_loss += loss
            scheduler.step()
            time_delta = time.time() - start

            self.model.eval()
            with torch.no_grad():
                for i, (x, y) in enumerate(self.val_dataloader):
                    if self.is_cuda:
                        x, y = x.cuda(), y.cuda()
                    logits = self.model(x)
                    loss = self.loss(logits, y)

                    val_loss += loss

            train_loss /= len(self.train_dataloader)
            val_loss /= len(self.val_dataloader)

            self._summarize(epoch + 1, train_loss, val_loss, time_delta)

            if save_every is not None and (epoch + 1) % save_every == 0:
                tag = epoch + 1
                self.save_checkpoint(tag=f'{tag}')

            if self.writer is not None:
                self.writer.add_scalar(f'{self.phase}: train loss', train_loss,
                                       epoch + 1)
                self.writer.add_scalar(f'{self.phase}: val loss', val_loss,
                                       epoch + 1)

            if self.verbose:
                out_str = f'epoch: {epoch}, train loss: {train_loss}, '
                out_str += f'val loss: {val_loss}, '
                out_str += f'time: {time_delta}'
                print(out_str)
