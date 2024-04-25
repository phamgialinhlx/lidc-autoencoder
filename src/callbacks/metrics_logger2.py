from typing import Any, Dict
import torch
from torch import Tensor
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from einops import rearrange
from pytorch_msssim import SSIM, MS_SSIM
from src.models import VQGAN, MultiheadVQGAN
from src.models.vq_gan_3d_seg_head import VQGANSegHead

class MetricsLogger(Callback):
    def __init__(
            self,
            ssim: SSIM | None = None,
            msssim: MS_SSIM | None = None,
            mean: float = 0.5,
            std: float = 0.5
    ):
        super().__init__()

        self.ssim = ssim
        self.ssim_storages = []
        self.msssim = msssim
        self.msssim_storages = []

        self.mean = mean
        self.std = std

    # def on_train_epoch_start(self, trainer: Trainer,
    #                          pl_module: LightningModule) -> None:
    #     self.reset_metrics()

    # def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule,
    #                        outputs: STEP_OUTPUT, batch: Any,
    #                        batch_idx: int) -> None:
    #     reals, conds = batch
    #     fakes = self.get_sample(pl_module, reals, conds)
    #     self.update_metrics(reals, fakes, device=pl_module.device)

    # def on_train_epoch_end(self, trainer: Trainer,
    #                        pl_module: LightningModule) -> None:
    #     self.log_metrics(pl_module, mode='train')

    def on_validation_epoch_start(self, trainer: Trainer,
                                  pl_module: LightningModule) -> None:
        self.reset_metrics()

    def on_validation_batch_end(self, trainer: Trainer,
                                pl_module: LightningModule,
                                outputs, batch: Any,
                                batch_idx: int) -> None:
        reals = batch['segmentation']
        conds = None # batch['label']
        fakes = self.get_sample(pl_module, batch, conds)
        self.update_metrics(reals, fakes, device=pl_module.device)

    def on_validation_epoch_end(self, trainer: Trainer,
                                pl_module: LightningModule) -> None:
        self.log_metrics(pl_module, mode='val')

    def on_test_epoch_start(self, trainer: Trainer,
                            pl_module: LightningModule) -> None:
        self.reset_metrics()

    def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule,
                          outputs, batch: Any,
                          batch_idx: int) -> None:
        # reals, conds = batch
        reals = batch['segmentation']
        conds = None # batch['label']
        fakes = self.get_sample(pl_module, batch, conds)
        self.update_metrics(reals, fakes, device=pl_module.device)

    def on_test_epoch_end(self, trainer: Trainer,
                          pl_module: LightningModule) -> None:
        self.log_metrics(pl_module, mode='test')

    def get_sample(self,
                   pl_module: LightningModule,
                   reals: Tensor | None = None,
                   conds: Dict[str, Tensor] = None):
        if isinstance(pl_module, VQGAN) or isinstance(pl_module, MultiheadVQGAN) or isinstance(pl_module, VQGANSegHead):
            if pl_module.use_ema:
                with pl_module.ema_scope():
                    _, _, reals, fakes = pl_module(reals, log_image=True)
            else:
                _, _, reals, fakes = pl_module(reals, log_image=True)
        else:
            raise NotImplementedError('this module is not Implemented')

        return fakes

    def reset_metrics(self):
        self.ssim_storages = []
        self.msssim_storages = []

    def update_metrics(self, reals: Tensor, fakes: Tensor,
                       device: torch.device):
        # convert range (-1, 1) to (0, 1)
        reals = (reals * self.std + self.mean).clamp(0, 1)
        fakes = (fakes * self.std + self.mean).clamp(0, 1)
        reals = rearrange(reals[0], 'c n h w  -> n c h w')
        fakes = rearrange(fakes[0], 'c n h w  -> n c h w')

        # update
        if self.ssim is not None:
            self.ssim.to(device)
            self.ssim_storages.append(self.ssim(reals, fakes))
            self.ssim.to('cpu')

        if self.msssim is not None:
            self.msssim.to(device)
            self.msssim_storages.append(self.msssim(reals, fakes))
            self.msssim.to('cpu')

    def log_metrics(self, pl_module: LightningModule, mode: str):
        if self.ssim is not None:
            self.ssim.to(pl_module.device)
            ssim_score = torch.stack(self.ssim_storages).mean()
            pl_module.log(mode + '/ssim',
                          ssim_score,
                          on_step=False,
                          on_epoch=True,
                          prog_bar=False,
                          sync_dist=True)
            self.ssim.to('cpu')

        if self.msssim is not None:
            self.msssim.to(pl_module.device)
            msssim_score = torch.stack(self.msssim_storages).mean()
            pl_module.log(mode + '/msssim',
                          msssim_score,
                          on_step=False,
                          on_epoch=True,
                          prog_bar=False,
                          sync_dist=True)
            self.msssim.to('cpu')
