import torch
from torch import Tensor
import wandb
from torchvision.utils import make_grid
import lightning as pl
from pytorch_lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

class LogMetricsCallback(Callback):
    def __init__(
            self,
            ssim=None,
            psnr=None,
            fid=None,
            inception_score=None,
            mean: float = 0.5,
            std: float = 0.5,
            seg_head: bool = False,
            input_key: str = "segmentation"
    ):
        super().__init__()

        self.ssim = ssim
        self.psnr = psnr
        self.fid = fid
        self.inception_score = inception_score
        self.mean = mean
        self.std = std
        self.seg_head = seg_head
        self.input_key = input_key

    @torch.no_grad()
    def reset_metrics(self):
        if self.ssim is not None:
            self.ssim.reset()

        if self.psnr is not None:
            self.psnr.reset()

        if self.fid is not None:
            self.fid.reset()

        if self.inception_score is not None:
            self.inception_score.reset()

    def on_validation_epoch_start(self, trainer: Trainer,
                                  pl_module: LightningModule) -> None:
        self.reset_metrics()

    def on_validation_batch_end(self, trainer: Trainer,
                                pl_module: LightningModule,
                                outputs, batch,
                                batch_idx: int) -> None:
        images = batch[self.input_key]
        image_outputs = self.get_sample(pl_module, images)
        if self.seg_head:
            images_seg = batch['segmentation']
            self.update_metrics(images_seg, image_outputs, device=pl_module.device)
        else:
            self.update_metrics(images, image_outputs, device=pl_module.device)

    def on_validation_epoch_end(self, trainer: Trainer,
                                pl_module: LightningModule) -> None:
        self.log_metrics(pl_module, mode='val')

    def on_test_epoch_start(self, trainer: Trainer,
                            pl_module: LightningModule) -> None:
        self.reset_metrics()

    def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule,
                          outputs, batch,
                          batch_idx: int) -> None:
        images = batch[self.input_key]
        image_outputs = self.get_sample(pl_module, images)
        if self.seg_head:
            images_seg = batch['segmentation']
            self.update_metrics(images_seg, image_outputs, device=pl_module.device)
        else:
            self.update_metrics(images, image_outputs, device=pl_module.device)

    def on_test_epoch_end(self, trainer: Trainer,
                          pl_module: LightningModule) -> None:
        self.log_metrics(pl_module, mode='test')

    def update_metrics(self, reals: Tensor, fakes: Tensor,
                       device: torch.device):
        # convert range (-1, 1) to (0, 1)
        fakes = (fakes * self.std + self.mean).clamp(0, 1)
        reals = (reals * self.std + self.mean).clamp(0, 1)

        # update
        if self.ssim is not None:
            self.ssim.to(device)
            self.ssim.update(fakes, reals)
            self.ssim.to('cpu')

        if self.psnr is not None:
            self.psnr.to(device)
            self.psnr.update(fakes, reals)
            self.psnr.to('cpu')

        # gray image
        if reals.shape[1] == 1:
            reals = torch.cat([reals, reals, reals], dim=1)
            fakes = torch.cat([fakes, fakes, fakes], dim=1)

        reals = torch.nn.functional.interpolate(reals,
                                                size=(299, 299),
                                                mode='bilinear')
        fakes = torch.nn.functional.interpolate(fakes,
                                                size=(299, 299),
                                                mode='bilinear')

        if self.fid is not None:
            self.fid.to(device)
            self.fid.update(reals, real=True)
            self.fid.update(fakes, real=False)
            self.fid.to('cpu')

        if self.inception_score is not None:
            self.inception_score.to(device)
            self.inception_score.update(fakes)
            self.inception_score.to('cpu')

    def get_sample(self,
                   pl_module: LightningModule,
                   reals: Tensor | None = None,
                   conds = None):
        fakes = pl_module.log_image(reals)
        return fakes

    def log_metrics(self, pl_module: LightningModule, mode: str):
        if self.ssim is not None:
            self.ssim.to(pl_module.device)
            pl_module.log(mode + '/ssim',
                          self.ssim.compute(),
                          on_step=False,
                          on_epoch=True,
                          prog_bar=False,
                          sync_dist=True)
            self.ssim.to('cpu')

        if self.psnr is not None:
            self.psnr.to(pl_module.device)
            pl_module.log(mode + '/psnr',
                          self.psnr.compute(),
                          on_step=False,
                          on_epoch=True,
                          prog_bar=False,
                          sync_dist=True)
            self.psnr.to('cpu')

        if self.fid is not None:
            self.fid.to(pl_module.device)
            pl_module.log(mode + '/fid',
                          self.fid.compute(),
                          on_step=False,
                          on_epoch=True,
                          prog_bar=False,
                          sync_dist=True)
            self.fid.to('cpu')

        if self.inception_score is not None:
            self.inception_score.to(pl_module.device)
            mean, std = self.inception_score.compute()
            range = {
                'min': mean - std,
                'max': mean + std,
            }

            pl_module.log(mode + '/is',
                          range,
                          on_step=False,
                          on_epoch=True,
                          prog_bar=False,
                          sync_dist=True)
            self.inception_score.to('cpu')
