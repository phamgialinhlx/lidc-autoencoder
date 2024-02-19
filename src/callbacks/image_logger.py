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
    def __init__(self, max_images, clamp=True, local=False):
        super().__init__()
        self.max_images = max_images
        self.clamp = clamp
        self.local = local
        self.batch = None

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch):
        root = os.path.join(save_dir, "images", split)
        # print(root)
        #mean = images.pop('mean_org')
        #mean = mean[(None,)*3].swapaxes(0, -1)
        #std = images.pop('std_org')
        #std = std[(None,)*3].swapaxes(0, -1)
        grids = []
        for k in images:
            images[k] = (images[k] + 1.0) * 127.5  # std + mean
            images[k] = images[k].clamp(0, 255)
            # images[k] = images[k].to(torch.uint8)
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            grid = grid
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid).astype(np.uint8)
            grids.append(grid)
            if self.local:
                filename = "{}_gs-{:06}_e-{:06}.png".format(
                    k,
                    global_step,
                    current_epoch)
                path = os.path.join(root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                Image.fromarray(grid).save(path)
        return grids
        
    def log_img(self, pl_module, split="train"):
        if (hasattr(pl_module, "log_images") and
            callable(pl_module.log_images) and
            self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(self.batch, split=split)
            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()

            grids = self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch)

            caption = [split + "_" + c for c in ["inputs", "reconstruct"]]
            pl_module.logger.experiment.log(
                {
                    "image": [
                        wandb.Image(grids[0]),
                        wandb.Image(grids[1]),
                    ],
                    "caption": caption,
                }
            )
            if is_train:
                pl_module.train()

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch: Any, batch_idx: int
    ) -> None:
        if self.batch is None:
            self.batch = batch
        else:
            # 70% get the new batch
            if np.random.rand() > 0.7:
                self.batch = batch
    
    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self.batch is None:
            self.batch = batch
        else:
            # 70% get the new batch
            if np.random.rand() > 0.7:
                self.batch = batch

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.log_img(pl_module, split="val")

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.log_img(pl_module, split="val")
    
