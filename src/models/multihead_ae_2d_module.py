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
from src.models.components.loss_function.lossbinary import LossBinary
from src.models.components.loss_function.lovasz_loss import BCE_Lovasz

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.components.vq_gan_2d.diffusionmodules import Encoder, Decoder
from src.models.components.vq_gan_2d.distributions import DiagonalGaussianDistribution
from src.models.components.vq_gan_2d.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from src.utils.ema import LitEma

class MultiheadVQGAN2d(LightningModule):
    def __init__(
        self,
        embed_dim,
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
        segmentation_decoder = None,
        segmentation_criterion = None,
        classifier_head = None,
        clasification_criterion = None,
        use_same_optimizer = False
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.automatic_optimization = False
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.image_key = image_key
        self.encoder = Encoder(**self.hparams.autoencoderconfig)
        self.decoder = Decoder(**self.hparams.autoencoderconfig)
        self.loss = loss
        self.quantize = VectorQuantizer(
            n_embed,
            embed_dim,
            beta=0.25,
            remap=remap,
            sane_index_shape=sane_index_shape,
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

        self.use_same_optimizer = use_same_optimizer
        self.segmentation_decoder = segmentation_decoder
        self.segmentation_criterion = segmentation_criterion
        self.classifier_head = classifier_head
        self.clasification_criterion = clasification_criterion
        self.bn = nn.BatchNorm3d(self.hparams.autoencoderconfig["z_channels"])

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
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward_clasification(self, batch, key="data"):
        x = batch[key]
        y = batch['label']
        logits = self.classifier_head(self.bn(self.encoder(x)))
        loss = self.clasification_criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def forward_segmentation(self, batch, key="data"):
        x = batch[key]
        y = batch['mask'].long()
        label = batch['label']
        if isinstance(self.segmentation_criterion, (LossBinary, BCE_Lovasz)):
            cnt1 = (y == 1).sum().item()  # count number of class 1 in image
            cnt0 = y.numel() - cnt1
            if cnt1 != 0:
                BCE_pos_weight = torch.FloatTensor([1.0 * cnt0 / cnt1]).to(device=self.device)
            else:
                BCE_pos_weight = torch.FloatTensor([1.0]).to(device=self.device)
            BCE_pos_weight = torch.FloatTensor([50.0]).to(device=self.device)
            self.segmentation_criterion.update_pos_weight(pos_weight=BCE_pos_weight)

        logits = self.segmentation_decoder(self.encoder, x)
        if self.segmentation_decoder.n_classes == 2:
            loss = self.segmentation_criterion(logits, y.squeeze(1))
            preds = torch.argmax(logits, dim=1).unsqueeze(0)
        else:
            loss = self.segmentation_criterion(logits, y.float())
            preds = torch.sigmoid(logits)
            preds[preds >= 0.5] = 1
            preds[preds < 0.5] = 0
        # Code to try to fix CUDA out of memory issues
        del x
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        return loss, preds, y

    def forward(self, input, return_pred_indices=False):
        quant, diff, (_, _, ind) = self.encode(input)
        dec = self.decode(quant)
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff

    def training_step(self, batch, batch_idx):
        # https://github.com/pytorch/pytorch/issues/37142
        # try not to fool the heuristics
        x = batch['segmentation']
        xrec, qloss, ind = self(x, return_pred_indices=True)

        opt_list = self.optimizers()
        opt_ae = opt_list[0]
        opt_disc = opt_list[1]

        if self.use_same_optimizer:
            opt_ds = opt_list[2]
        else:
            if len(opt_list) == 3:
                if self.segmentation_decoder is not None:
                    opt_seg = opt_list[2]
                else:
                    opt_cls = opt_list[2]
            elif len(opt_list) == 4:
                opt_seg = opt_list[2]
                opt_cls = opt_list[3]

        # autoencode
        aeloss, log_dict_ae = self.loss(
            qloss,
            x,
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
            x,
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

        if self.use_same_optimizer:
            opt_ds.zero_grad()
            seg_loss, seg_preds, seg_targets = self.forward_segmentation(batch, key="segmentation")
            cls_loss, cls_preds, cls_targets = self.forward_clasification(batch, key="segmentation")
            ds_loss = seg_loss + cls_loss
            self.manual_backward(ds_loss)
            opt_ds.step()
            self.log("train/ds_loss", ds_loss, prog_bar=True,
                     logger=True, on_step=True, on_epoch=True)
        else:
            if self.segmentation_decoder is not None:
                opt_seg.zero_grad()
                seg_loss, seg_preds, seg_targets = self.forward_segmentation(batch, key="segmentation")
                self.manual_backward(seg_loss)
                opt_seg.step()
            if self.classifier_head is not None:
                opt_cls.zero_grad()
                cls_loss, cls_preds, cls_targets = self.forward_clasification(batch, key="segmentation")
                self.manual_backward(cls_loss)
                opt_cls.step()

        if self.classifier_head is not None and self.segmentation_decoder is not None:
            return {"seg_loss": seg_loss, "seg_preds": seg_preds, "seg_targets": seg_targets,
                    "cls_loss": cls_loss, "cls_preds": cls_preds, "cls_targets": cls_targets}
        elif self.classifier_head is None and self.segmentation_decoder is not None:
            return {"seg_loss": seg_loss, "seg_preds": seg_preds, "seg_targets": seg_targets}

    def validation_step(self, batch, batch_idx):
        if self.use_ema:
            with self.ema_scope():
                log_dict_ema = self._validation_step(batch, batch_idx)
        else:
            log_dict = self._validation_step(batch, batch_idx)
        if self.segmentation_decoder is not None:
            seg_loss, seg_preds, seg_targets = self.forward_segmentation(batch, key="segmentation")
        if self.classifier_head is not None:
            cls_loss, cls_preds, cls_targets = self.forward_clasification(batch, key="segmentation")
        if self.segmentation_decoder is not None and self.classifier_head is not None:
            return {"seg_loss": seg_loss, "seg_preds": seg_preds, "seg_targets": seg_targets,
                    "cls_loss": cls_loss, "cls_preds": cls_preds, "cls_targets": cls_targets}
        elif self.classifier_head is None and self.segmentation_decoder is not None:
            return {"seg_loss": seg_loss, "seg_preds": seg_preds, "seg_targets": seg_targets}

    def _validation_step(self, batch, batch_idx, suffix=""):
        x = batch['segmentation']
        xrec, qloss, ind = self(x, return_pred_indices=True)
        aeloss, log_dict_ae = self.loss(
            qloss,
            x,
            xrec,
            0,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val" + suffix,
            # predicted_indices=ind,
        )

        discloss, log_dict_disc = self.loss(
            qloss,
            x,
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
                if self.segmentation_decoder is not None:
                    seg_loss, seg_preds, seg_targets = self.forward_segmentation(batch, key="segmentation")
                if self.classifier_head is not None:
                    cls_loss, cls_preds, cls_targets = self.forward_clasification(batch, key="segmentation")
        else:
            log_dict = self._test_step(batch, batch_idx)
            if self.segmentation_decoder is not None:
                seg_loss, seg_preds, seg_targets = self.forward_segmentation(batch, key="segmentation")
            if self.classifier_head is not None:
                cls_loss, cls_preds, cls_targets = self.forward_clasification(batch, key="segmentation")

        if self.segmentation_decoder is not None and self.classifier_head is not None:
            return {"seg_loss": seg_loss, "seg_preds": seg_preds, "seg_targets": seg_targets,
                    "cls_loss": cls_loss, "cls_preds": cls_preds, "cls_targets": cls_targets}
        elif self.classifier_head is None and self.segmentation_decoder is not None:
            return {"seg_loss": seg_loss, "seg_preds": seg_preds, "seg_targets": seg_targets}

    def _test_step(self, batch, batch_idx, suffix=""):
        x = batch['segmentation']
        xrec, qloss, ind = self(x, return_pred_indices=True)
        aeloss, log_dict_ae = self.loss(
            qloss,
            x,
            xrec,
            0,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="test" + suffix,
            # predicted_indices=ind,
        )

        discloss, log_dict_disc = self.loss(
            qloss,
            x,
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

        opt_list = [opt_ae, opt_disc]
        if self.use_same_optimizer:
            opt_ds = torch.optim.Adam(list(self.segmentation_decoder.parameters()) + \
                                      list(self.encoder.parameters()) + \
                                      list(self.classifier_head.parameters()),
                                      lr=0.001, betas=(0.5, 0.9))
            opt_list.append(opt_ds)
        else:
            if self.segmentation_decoder is not None:
                opt_seg = torch.optim.Adam(list(self.segmentation_decoder.parameters()) + \
                                        list(self.encoder.parameters()), lr=0.001, betas=(0.5, 0.9))
                opt_list.append(opt_seg)
            if self.classifier_head is not None:
                opt_cls = torch.optim.Adam(list(self.classifier_head.parameters()) + \
                                        list(self.encoder.parameters()), lr=0.001, betas=(0.5, 0.9))
                opt_list.append(opt_cls)

        return (opt_list, [])

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
    cfg.model.autoencoderconfig.channel_multipliers = [1, 1, 2, 4] # 73165400 params
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    input = torch.randn(2, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
    print("Number of params: ", sum(p.numel() for p in model.parameters()))

    def encode_check(model, img):
        x = model.encoder.conv_in(img)

        hs = [x]
        for i_level in range(model.encoder.num_resolutions):
            for i_block in range(model.encoder.num_res_blocks):
                h = model.encoder.down[i_level].block[i_block](hs[-1])
                if len(model.encoder.down[i_level].attn) > 0:
                    h = model.encoder.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != model.encoder.num_resolutions - 1:
                hs.append(model.encoder.down[i_level].downsample(hs[-1]))

        # middle
        x = hs[-1]

        # Final ResNet blocks with attention
        x = model.encoder.mid.block_1(x)
        x = model.encoder.mid.attn_1(x)
        x = model.encoder.mid.block_2(x)

        # Normalize and map to embedding space
        x = model.encoder.norm_out(x)
        x = model.encoder.swish(x)
        print("Encoder pre conv_out:", x.shape)
        x = model.encoder.conv_out(x)
        return x
    # Encode
    h = encode_check(model, input)
    print("Encoder output:", h.shape)
    h = model.quant_conv(h)
    print("Quant conv output:", h.shape)
    quant, emb_loss, info = model.quantize(h)
    print("Quantized output:", quant.shape)
    # Decode
    dec = model.decode(quant)
    print(dec.shape)


if __name__ == "__main__":
    main()
