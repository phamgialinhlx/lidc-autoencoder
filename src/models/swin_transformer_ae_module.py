from typing import List, Optional, Tuple, Any
from contextlib import contextmanager
import torch
from torch import nn
from torchmetrics import MaxMetric, MeanMetric
from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import hydra
from omegaconf import DictConfig
from typing_extensions import Final
from monai.utils import ensure_tuple_rep, look_up_option, optional_import

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.components.vq_gan_2d.diffusionmodules import Encoder, Decoder
from src.models.components.vq_gan_2d.distributions import DiagonalGaussianDistribution
from src.models.components.vq_gan_2d.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from src.models.components.monai.swin_unetr import SwinTransformer, MERGING_MODE
from src.utils.ema import LitEma

class SwinVQGAN(LightningModule):

    patch_size: Final[int] = 2

    def __init__(
        self,
        embed_dim,
        encoderconfig,
        autoencoderconfig,
        loss,
        n_embed: int = 16384,
        image_key=0,
        ckpt_path=None,
        ignore_keys=[],
        colorize_nlabels=None,
        monitor=None,
        batch_resize_range=None,
        scheduler_config=None,
        lr: float = 4.5e-6,
        lr_g_factor=1.0,
        remap=None,
        sane_index_shape=False,  # tell vector quantizer to return indices as bhw
        use_ema=False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.automatic_optimization = False
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.image_key = image_key
        
        _img_size = ensure_tuple_rep(encoderconfig.img_size, encoderconfig.spatial_dims)
        _patch_sizes = ensure_tuple_rep(self.patch_size, encoderconfig.spatial_dims)
        _window_size = ensure_tuple_rep(7, encoderconfig.spatial_dims)
        
        self.encoder_normalize = encoderconfig.normalize

        self.encoder = SwinTransformer(
            in_chans=encoderconfig.in_channels,
            embed_dim=encoderconfig.feature_size,
            window_size=_window_size,
            patch_size=_patch_sizes,
            depths=encoderconfig.depths,
            num_heads=encoderconfig.num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=encoderconfig.drop_rate,
            attn_drop_rate=encoderconfig.attn_drop_rate,
            drop_path_rate=encoderconfig.dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=encoderconfig.use_checkpoint,
            spatial_dims=encoderconfig.spatial_dims,
            downsample=look_up_option(encoderconfig.downsample, MERGING_MODE) if isinstance(encoderconfig.downsample, str) else encoderconfig.downsample,
            use_v2=encoderconfig.use_v2,
        )

        self.decoder = Decoder(**self.hparams.autoencoderconfig)
        self.loss = loss
        self.quantize = VectorQuantizer(
            n_embed,
            embed_dim,
            beta=0.25,
            remap=remap,
            sane_index_shape=sane_index_shape,
        )
        self.quant_conv_module = nn.Sequential(
            torch.nn.ConvTranspose2d(encoderconfig.feature_size * 16, encoderconfig.feature_size * 8, kernel_size=2, stride=2),
            torch.nn.ConvTranspose2d(encoderconfig.feature_size * 8, encoderconfig.feature_size * 4, kernel_size=2, stride=2),
            torch.nn.ConvTranspose2d(encoderconfig.feature_size * 4, embed_dim, kernel_size=2, stride=2),
        )
        self.quant_conv = torch.nn.Conv2d(self.hparams.autoencoderconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, self.hparams.autoencoderconfig["z_channels"], 1)
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) is int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        self.batch_resize_range = batch_resize_range
        if self.batch_resize_range is not None:
            print(
                f"{self.__class__.__name__}: Using per-batch resizing in range {batch_resize_range}."
            )

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.scheduler_config = scheduler_config
        self.lr_g_factor = lr_g_factor
        self.learning_rate = lr

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(
            f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def encode(self, x):
        h = self.encoder(x)[-1]
        h = self.quant_conv_module(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv_module(h)
        return h

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, return_pred_indices=False):
        quant, diff, (_, _, ind) = self.encode(input)
        dec = self.decode(quant)
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff

    def training_step(self, batch, batch_idx):
        # https://github.com/pytorch/pytorch/issues/37142
        # try not to fool the heuristics
        x_seg = batch['segmentation']
        xrec, qloss, ind = self(x_seg, return_pred_indices=True)

        opt_ae, opt_disc = self.optimizers()

        # autoencode
        aeloss, log_dict_ae = self.loss(
            qloss,
            x_seg,
            xrec,
            0,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="train",
            # predicted_indices=ind,
        )

        self.log_dict(
            log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True
        )

        opt_ae.zero_grad()
        self.manual_backward(aeloss)
        opt_ae.step()

        # discriminator
        discloss, log_dict_disc = self.loss(
            qloss,
            x_seg,
            xrec,
            1,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="train",
        )
        self.log_dict(
            log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True
        )

        opt_disc.zero_grad()
        self.manual_backward(discloss)
        opt_disc.step()

    def validation_step(self, batch, batch_idx):
        if self.use_ema:
            with self.ema_scope():
                log_dict_ema = self._validation_step(batch, batch_idx)
        else:
            log_dict = self._validation_step(batch, batch_idx)

    def _validation_step(self, batch, batch_idx, suffix=""):
        x_seg = batch['segmentation']
        xrec, qloss, ind = self(x_seg, return_pred_indices=True)
        aeloss, log_dict_ae = self.loss(
            qloss,
            x_seg,
            xrec,
            0,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val" + suffix,
            # predicted_indices=ind,
        )

        discloss, log_dict_disc = self.loss(
            qloss,
            x_seg,
            xrec,
            1,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val" + suffix,
            # predicted_indices=ind,
        )

        self.log(f"val{suffix}/rec_loss", log_dict_ae[f"val{suffix}/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def test_step(self, batch, batch_idx) -> None:
        if self.use_ema:
            with self.ema_scope():
                log_dict_ema = self._test_step(batch, batch_idx, suffix="")
        else:
            log_dict = self._test_step(batch, batch_idx)

    def _test_step(self, batch, batch_idx, suffix=""):
        x_seg = batch['segmentation']
        xrec, qloss, ind = self(x_seg, return_pred_indices=True)
        aeloss, log_dict_ae = self.loss(
            qloss,
            x_seg,
            xrec,
            0,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="test" + suffix,
            # predicted_indices=ind,
        )

        discloss, log_dict_disc = self.loss(
            qloss,
            x_seg,
            xrec,
            1,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="test" + suffix,
            # predicted_indices=ind,
        )

        self.log(f"test{suffix}/rec_loss", log_dict_ae[f"test{suffix}/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr_d = self.learning_rate
        lr_g = self.lr_g_factor * self.learning_rate
        opt_ae = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.quantize.parameters())
            + list(self.quant_conv.parameters())
            + list(self.post_quant_conv.parameters()),
            lr=lr_g,
            betas=(0.5, 0.9),
        )
        opt_disc = torch.optim.Adam(
            self.loss.discriminator.parameters(), lr=lr_d, betas=(0.5, 0.9)
        )

        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    "scheduler": LambdaLR(opt_ae, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1,
                },
                {
                    "scheduler": LambdaLR(opt_disc, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1,
                },
            ]
            return [opt_ae, opt_disc], scheduler
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_image(
        self,
        images, 
        device: torch.device = torch.device('cpu'),
    ):
        # # Encode
        # h = self.encoder(images)
        # h = self.quant_conv(h)
        # quant, emb_loss, info = self.quantize(h)
        # # Decode
        # quant = self.post_quant_conv(quant)
        # dec = self.decoder(quant)
        if self.use_ema:
            with self.ema_scope():
                dec, _, = self.forward(images)
        else:
            dec, _, = self.forward(images)
        return dec
@hydra.main(
    version_base="1.3", config_path="../../configs", config_name="train.yaml"
)
def main(cfg: DictConfig) -> Optional[float]:
    print(f"Instantiating model <{cfg.model._target_}>")
    IMG_SIZE = 128
    IMG_CHANNELS = 1
    cfg.model.autoencoderconfig.channels = IMG_SIZE
    cfg.model.autoencoderconfig.img_channels = IMG_CHANNELS
    # cfg.model.autoencoderconfig.channel_multipliers = [1, 2, 4] # 72378612 params
    cfg.model.autoencoderconfig.channel_multipliers = [1, 2, 4] # 73165400 params
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    input = torch.randn(2, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
    print("Number of params: ", sum(p.numel() for p in model.parameters()))

    from IPython import embed; embed()
    # Encode
    h = model.encoder(input, model.encoder_normalize)[-1]
    print("Encoder output:", h.shape)
    h = model.quant_conv_module(h)
    print("Quant conv output:", h.shape)
    quant, emb_loss, info = model.quantize(h)
    print("Quantized output:", quant.shape)
    # Decode
    dec = model.decode(quant)
    print(dec.shape)


if __name__ == "__main__":
    main()
