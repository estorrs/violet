import json
import os
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
    def __init__(self, model, num_classes):
        super().__init__()

        self.vit = model
        self.head = nn.Linear(self.vit.num_features, num_classes)

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
                 max_lr=1e-4, verbose=True, summary=None):
        self.model = model
        self.max_lr = max_lr
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

    def _get_optimizer(self, epochs):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.max_lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=self.max_lr,
            steps_per_epoch=len(self.train_dataloader), epochs=epochs)

        return opt, scheduler

    def _summarize(self, epoch, train_loss, val_loss):
        if 'training' not in self.summary:
            self.summary['training'] = {}
        if self.phase not in self.summary['training']:
            self.summary['training'][self.phase] = {}
        self.summary['training'][self.phase][epoch] = {
            'train_loss': train_loss.tolist(),
            'val_loss': val_loss.tolist()
        }

    def unfreeze_vit(self):
        self.model.unfreeze_vit()
        self.phase = '2.finetune_unfrozen_vit'

    def save_checkpoint(self, tag=''):
        chkpt = f'{self.phase}{tag}.pt'
        save_path = os.path.join(self.checkpoint_dir, chkpt)
        torch.save(self.model.state_dict(), save_path)

        return save_path

    def save_final(self):
        save_path = os.path.join(self.checkpoint_dir, 'final.pt')
        torch.save(self.model.state_dict(), save_path)
        summary_path = os.path.join(self.run_dir, 'summary.json')
        json.dump(self.summary, open(summary_path, 'w'))

        return save_path, summary_path

    def fit(self, epochs):
        opt, scheduler = self._get_optimizer(epochs)

        for epoch in range(epochs):
            train_loss, val_loss = 0., 0.

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

            self._summarize(epoch + 1, train_loss, val_loss)

            if self.writer is not None:
                self.writer.add_scalar(f'{self.phase}: train loss', train_loss,
                                       epoch + 1)
                self.writer.add_scalar(f'{self.phase}: val loss', val_loss,
                                       epoch + 1)

            if self.verbose:
                print(f'epoch: {epoch}, train loss: {train_loss}, val loss: {val_loss}')
