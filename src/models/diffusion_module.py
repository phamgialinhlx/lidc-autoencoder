from typing import Any, Tuple, Optional, Dict
from contextlib import contextmanager
import torch
from tqdm import tqdm
from lightning import LightningModule
import hydra
from torch import nn, Tensor
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.optim import Optimizer, lr_scheduler

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.ema import LitEma
from src.models.vq_gan_2d_module import VQGAN
from src.models.swin_transformer_ae_module import SwinVQGAN
from src.models.components.diffusion.sampler import BaseSampler
from src.models.components.diffusion.sampler.ddpm import DDPMSampler

def load_autoencoder(ckpt_path, map_location="cuda", disable_decoder=False, eval=True):
    # Attempt to load the SwinVQGAN model as a fallback
    try:
        ae = SwinVQGAN.load_from_checkpoint(checkpoint_path=ckpt_path, map_location=map_location)
        if ae.use_ema:
            ae.model_ema.store(ae.parameters())
            ae.model_ema.copy_to(ae)
        
        # Disable the decoder if requested
        if disable_decoder:
            ae.decoder = None

    except Exception as e:
        print(f"Failed to load SwinVQGAN from {ckpt_path}: {e}")
        try:
            ae = VQGAN.load_from_checkpoint(checkpoint_path=ckpt_path, map_location=map_location)
            if ae.use_ema:
                ae.model_ema.store(ae.parameters())
                ae.model_ema.copy_to(ae)
            
            # Disable the decoder if requested
            if disable_decoder:
                ae.decoder = None
        except Exception as e:
            print(f"Failed to load VQGAN from {ckpt_path}: {e}")
            return None
    if eval:
        ae.eval()
        ae.freeze()
    return ae

class DiffusionModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        autoencoder_ckpt_path,
        optimizer: Optimizer,
        scheduler: lr_scheduler,
        use_ema: bool = False,
        sampler: BaseSampler = DDPMSampler(),
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.autoencoder = load_autoencoder(autoencoder_ckpt_path)

        self.net = net
        self.sampler = sampler
        # exponential moving average
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.net)

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.net)

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.net.parameters())
            self.model_ema.copy_to(self.net)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.net.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def loss(self, x0: torch.Tensor, noise: Optional[torch.Tensor] = None):
        t = torch.randint(
            0,
            self.sampler.n_train_steps,
            [x0.shape[0]],
            device=x0.device,
        )
        xt, noise = self.sampler.step(x0, t, noise)
        eps_theta = self.net(xt, t)
        return F.mse_loss(noise, eps_theta)

    @torch.no_grad()
    def autoencoder_encode(self, x):

        if self.autoencoder is None:
            return x
        else:
            if isinstance(self.autoencoder, VQGAN):
                quant, emb_loss, info = self.autoencoder.encode(x)
                # normalize to -1 and 1
                # x = ((x - self.autoencoder.codebook.embeddings.min()) /
                #     (self.autoencoder.codebook.embeddings.max() -
                #     self.autoencoder.codebook.embeddings.min())) * 2.0 - 1.0
                return quant
            elif isinstance(self.autoencoder, SwinVQGAN):
                return self.autoencoder.encode(x)[0]
            else:
                return self.autoencoder.encode(x).sample()
            
    @torch.no_grad()
    def autoencoder_decode(self, xt):
        if self.autoencoder is None:
            return xt
        else:
            if isinstance(self.autoencoder, VQGAN):
                # denormalize TODO: Remove eventually
                # codebook_embeddings = self.autoencoder.quantize.embedding
                # xt = (((xt + 1.0) / 2.0) * (codebook_embeddings.max() -
                #                         codebook_embeddings.min())) + codebook_embeddings.min()

                xt = self.autoencoder.decode(xt)
                return xt
            elif isinstance(self.autoencoder, SwinVQGAN):
                return self.autoencoder.decode(xt)
            else:
                raise NotImplementedError("Not implemented for VQGAN")
            
    def model_step(self, batch: Any):
        # images, labels = batch
        images = batch['segmentation']
        latent = self.autoencoder_encode(images)
        return self.loss(latent)

    def training_step(self, batch: Any, batch_idx: int):
        loss = self.model_step(batch)
        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"])
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        loss = self.model_step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch: Any, batch_idx: int):
        loss = self.model_step(batch)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=optimizer, lr_lambda=self.hparams.scheduler.schedule
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    @torch.no_grad()
    def log_videos(
        self,
        xt,
        noise: torch.Tensor = None,
        repeat_noise: bool = False,
        cond: Tensor = None,
        device: torch.device = torch.device('cuda'),
        prog_bar: bool = True, 
        **kwargs
    ) -> Tensor:
        xt = xt['data']
        xt = self.autoencoder_encode(xt)
        xt = torch.randn(xt.shape).to(device)

        sample_steps = (
            tqdm(self.sampler.timesteps, desc="Sampling t")
            if prog_bar
            else self.sampler.timesteps
        )
        
        if self.use_ema:
            # generate sample by ema_model
            with self.ema_scope():
                for i, t in enumerate(sample_steps):
                    
                    t = torch.full((xt.shape[0],), t, device=device, dtype=torch.int64)
                    model_output = self.net(x=xt, time=t, cond=cond)
                    xt = self.sampler.reverse_step(
                        model_output, t, xt, noise, repeat_noise
                    )
        else:
            for i, t in enumerate(sample_steps):
                t = torch.full((xt.shape[0],), t, device=device, dtype=torch.int64)
                model_output = self.net(x=xt, time=t, cond=cond)
                xt = self.sampler.reverse_step(
                    model_output, t, xt, noise, repeat_noise
                )
        out_images = self.autoencoder_decode(xt)
        return {"generate": out_images}
    
    def log_image(
            self, 
            x, 
            noise: torch.Tensor = None,
            repeat_noise: bool = False,
            prog_bar: bool = True,
            cond: Tensor = None,
            device: torch.device = torch.device('cuda')):
        xt = self.autoencoder_encode(x)
        xt = torch.randn(xt.shape).to(device)

        sample_steps = (
            tqdm(self.sampler.timesteps, desc="Sampling t")
            if prog_bar
            else self.sampler.timesteps
        )
        
        if self.use_ema:
            # generate sample by ema_model
            with self.ema_scope():
                for i, t in enumerate(sample_steps):
                    t = torch.full((xt.shape[0],), t, device=device, dtype=torch.int64)
                    model_output = self.net(x=xt, timesteps=t, context=cond)
                    xt = self.sampler.reverse_step(
                        model_output, t, xt, noise, repeat_noise
                    )
        else:
            for i, t in enumerate(sample_steps):
                t = torch.full((xt.shape[0],), t, device=device, dtype=torch.int64)
                model_output = self.net(x=xt, timesteps=t, context=cond)
                xt = self.sampler.reverse_step(
                    model_output, t, xt, noise, repeat_noise
                )
        out_images = self.autoencoder_decode(xt)
        return out_images