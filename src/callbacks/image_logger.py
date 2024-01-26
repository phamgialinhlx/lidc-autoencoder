from typing import Any, Dict, Type

import os
import wandb
import torch
import torchvision
from PIL import Image

import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only

class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True, local=False):
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.log_steps = [
            2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.local = local

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        # print(root)
        #mean = images.pop('mean_org')
        #mean = mean[(None,)*3].swapaxes(0, -1)
        #std = images.pop('std_org')
        #std = std[(None,)*3].swapaxes(0, -1)
        grids = []
        for k in images:
            images[k] = (images[k] + 1.0) * 127.5  # std + mean
            torch.clamp(images[k], 0, 255)
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            grid = grid
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid).astype(np.uint8)
            grids.append(grid)
            if self.local:
                filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                    k,
                    global_step,
                    current_epoch,
                    batch_idx)
                path = os.path.join(root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                Image.fromarray(grid).save(path)
        return grids
        
    def log_img(self, pl_module, batch, batch_idx, split="train"):
        if (self.check_frequency(batch_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
            self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()

            grids = self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)
            
            pl_module.logger.experiment.log(
                {
                    "image": [
                        wandb.Image(grids[0]),
                        wandb.Image(grids[1]),
                    ],
                    "caption": ["inputs", "reconstruct"],
                }
            )
            if is_train:
                pl_module.train()

    def check_frequency(self, batch_idx):
        if (batch_idx % self.batch_freq) == 0 or (batch_idx in self.log_steps):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch: Any, batch_idx: int
    ) -> None:
        self.log_img(pl_module, batch, batch_idx, split="train")
    
    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        # print("validation epoch end")
        self.log_img(pl_module, batch, batch_idx, split="val")