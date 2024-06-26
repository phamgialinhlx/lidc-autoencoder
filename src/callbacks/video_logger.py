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

from src.models.diffusion_module import DiffusionModule

def video_grid(video, fname, nrow=None, fps=6, save_local=False):
    b, c, t, h, w = video.shape
    video = video.permute(0, 2, 3, 4, 1)
    video = video.clamp(0, 1)
    video = (video.cpu().numpy() * 255).astype('uint8')
    
    if nrow is None:
        nrow = math.ceil(math.sqrt(b))
    ncol = math.ceil(b / nrow)
    padding = 1
    video_grid = np.zeros((t, (padding + h) * nrow + padding,
                           (padding + w) * ncol + padding, c), dtype='uint8')
    for i in range(b):
        r = i // ncol
        c = i % ncol
        start_r = (padding + h) * r
        start_c = (padding + w) * c
        video_grid[:, start_r:start_r + h, start_c:start_c + w] = video[i]
    video = []
    for i in range(t):
        video.append(video_grid[i])
    imageio.mimsave(fname, video, fps=fps)
    return fname

class VideoLogger(Callback):
    def __init__(self, max_videos, clamp=True, local=False):
        super().__init__()
        self.max_videos = max_videos
        self.clamp = clamp
        self.local = local
        self.batch = None

    @rank_zero_only
    def log_local(self, save_dir, split, videos,
                  global_step, current_epoch):
        root = os.path.join(save_dir, "videos", split)
        # print(root)
        # mean = videos.pop('mean_org')
        # mean = mean[(None,)*4].swapaxes(0, -1)
        # std = videos.pop('std_org')
        # std = std[(None,)*4].swapaxes(0, -1)
        grids = []
        for k in videos:
            videos[k] = (videos[k] + 1.0) * 127.5  # std + mean
            torch.clamp(videos[k], 0, 255)
            videos[k] = videos[k] / 255.0
            grid = videos[k]
            filename = "{}_gs-{:06}_e-{:06}.mp4".format(
                k,
                global_step,
                current_epoch,
            )
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            fname = video_grid(grid, path, save_local=True)
            grids.append(fname)
        return grids

    def log_vid(self, pl_module, split="train"):
        if (hasattr(pl_module, "log_videos") and
            callable(pl_module.log_videos) and
            self.max_videos > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                videos = pl_module.log_videos(
                    self.batch, split=split)

            for k in videos:
                N = min(videos[k].shape[0], self.max_videos)
                videos[k] = videos[k][:N]
                if isinstance(videos[k], torch.Tensor):
                    videos[k] = videos[k].detach().cpu()

            fnames = self.log_local(pl_module.logger.save_dir, split, videos,
                           pl_module.global_step, pl_module.current_epoch)
            if is_train:
                pl_module.train()
            
            if isinstance(pl_module, DiffusionModule):
                pl_module.logger.experiment.log(
                    {
                        "video": [
                            wandb.Video(fnames[0]),
                        ],
                        "caption": [split + "_" + c for c in ["generate"]],
                    }
                )
            else:
                caption = [split + "_" + c for c in ["inputs", "reconstruct"]]
                pl_module.logger.experiment.log(
                    {
                        "video": [
                            wandb.Video(fnames[0]),
                            wandb.Video(fnames[1]),
                        ],
                        "caption": caption,
                    }
                )
            if not self.local:
                for fname in fnames:
                    os.remove(fname)

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
        self.log_vid(pl_module, split="val")

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.log_vid(pl_module, split="train")
    
