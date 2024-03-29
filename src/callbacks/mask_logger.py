from typing import Any, Dict, Type

import os
import wandb
import torch
import torchvision
from PIL import Image
import imageio

import math
import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only
from einops import rearrange
import torch.nn.functional as F


class MaskLogger(Callback):
    def __init__(self, max_videos, clamp=True, local=False):
        super().__init__()
        self.max_videos = max_videos
        self.clamp = clamp
        self.local = local
        self.batch = None

    def log_vid(self, pl_module, split="train"):

        if (self.max_videos > 0):
            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            root = os.path.join(pl_module.logger.save_dir, "videos", split)
            filename = "{}_gs-{:06}_e-{:06}.mp4".format(
                'video',
                pl_module.global_step,
                pl_module.current_epoch,
                )
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)

            x = self.batch['data']
            
            y = self.batch['mask'].int().squeeze(0)
            label = self.batch['label']
            if (hasattr(pl_module, "forward_segmentation")):
                _, pred, _ = pl_module.forward_segmentation(self.batch)
            else:
                pred = pl_module.forward(x)
            pred = torch.argmax(pred, dim=1).unsqueeze(0)
            # pred = pred >= 0.5
            pred = pred.squeeze(0)
            # pred = torch.argmax(pred, dim=1)
            x = (x + 1.0) * 127.5  # std + mean
            torch.clamp(x, 0, 255)
            
            x = x.squeeze(0)
            x = x.permute(1, 2, 3, 0)
            x = x.cpu().numpy()
            y[y == 1] = 255
            y = y.permute(1, 2, 3, 0)
            y = y.cpu().numpy()
            pred = pred.permute(1, 2, 3, 0)
            pred[pred == 1] = 255
            pred = pred.cpu().detach().numpy()

            frames = []
            t, h, w, c = x.shape
            # frame = torch.concat((x[0], y[0], pred[0]), dim=1)  # Concatenate images horizontally
            # frame = np.concatenate((x[0], y[0], pred[0]), axis=1)  # Concatenate images horizontally

            for i in range(t):
                # Assuming x, y, and pred are images represented as numpy arrays
                frame = np.concatenate((x[i], y[i], pred[i]), axis=1)  # Concatenate images horizontally
                frames.append(frame.astype(np.uint8))

            imageio.mimsave(path, frames, fps=6)
 
            if is_train:
                pl_module.train()
            
            pl_module.logger.experiment.log(
                {
                    "video_gt_pred": [
                        wandb.Video(path),
                    ],
                    "caption": [split + "_" + c for c in ["video"]],
                }
            )
            if not self.local:
                os.remove(path)

            if is_train:
                pl_module.train()

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch: Any, batch_idx: int
    ) -> None:
        if self.batch is None:
            self.batch = batch
        else:
            if self.batch['label'] == 0:
                self.batch = batch
            elif batch['label'] != 0:
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
        self.log_vid(pl_module, split="val")

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.log_vid(pl_module, split="train")
    
