from typing import Any
from contextlib import contextmanager
import torch
from lightning import LightningModule
import hydra
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.ema import LitEma
from src.models.components.vqgan import Encoder, Decoder, SamePadConv3d, Codebook, LPIPS, NLayerDiscriminator, NLayerDiscriminator3D
from src.models.components.vqgan.utils import shift_dim, adopt_weight, comp_getattr
from src.models.components.loss_function.lossbinary import LossBinary
from src.models.components.loss_function.lovasz_loss import BCE_Lovasz
from torchmetrics import Dice, JaccardIndex, MaxMetric, MeanMetric

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) + \
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss

def load_autoencoder(ckpt_path, map_location="cuda", disable_decoder=False, eval=True):
    try:
        ae = MultiheadVQGAN.load_from_checkpoint(ckpt_path, map_location=map_location)
        if ae.use_ema:
            ae.model_ema.store(ae.parameters())
            ae.model_ema.copy_to(ae)
        if disable_decoder:
            ae.decoder = None

    except Exception as e:
        print(f"Failed to load autoencoder from {ckpt_path}: {e}")
    if eval:
        ae.eval()
        ae.freeze()
    return ae

class MultiheadVQGAN(LightningModule):
    def __init__(self,
            embedding_dim: int = 256,
            n_codes: int = 2048,
            n_hiddens: int = 240,
            lr: float = 3e-4,
            image_channels: int = 1,
            downsample: Any = [4, 4, 4],
            disc_channels: int = 64,
            disc_layers: int = 3,
            discriminator_iter_start: int = 50000,
            disc_loss_type: str = "hinge",
            image_gan_weight: float = 1.0,
            video_gan_weight: float = 1.0,
            l1_weight: float = 4.0,
            gan_feat_weight: float = 0.0,
            perceptual_weight: float = 0.0,
            i3d_feat: bool = False,
            restart_thres: float = 1.0,
            no_random_restart: bool = False,
            norm_type: str = "group",
            padding_type: str = "replicate",
            num_groups: int = 32,
            use_ema: bool = True,
            segmentation_decoder = None,
            segmentation_criterion = None,
            classifier_head = None,
            clasification_criterion = None,
            use_same_optimizer = False):

        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.embedding_dim = embedding_dim
        self.n_codes = n_codes
        self.lr = lr
        self.discriminator_iter_start = discriminator_iter_start

        self.encoder = Encoder(
            n_hiddens,
            downsample,
            image_channels,
            norm_type,
            padding_type,
            num_groups,
        )
        self.decoder = Decoder(
            n_hiddens,
            downsample,
            image_channels,
            norm_type,
            num_groups
        )
        self.enc_out_ch = self.encoder.out_channels
        self.pre_vq_conv = SamePadConv3d(
            self.enc_out_ch,
            embedding_dim,
            1,
            padding_type=padding_type
        )
        self.post_vq_conv = SamePadConv3d(
            embedding_dim,
            self.enc_out_ch,
            1
        )

        self.codebook = Codebook(
            n_codes,
            embedding_dim,
            no_random_restart=no_random_restart,
            restart_thres=restart_thres)

        self.gan_feat_weight = gan_feat_weight
        # TODO: Changed batchnorm from sync to normal
        self.image_discriminator = NLayerDiscriminator(
            image_channels, disc_channels, disc_layers, norm_layer=nn.BatchNorm2d)
        self.video_discriminator = NLayerDiscriminator3D(
            image_channels, disc_channels, disc_layers, norm_layer=nn.BatchNorm3d)

        if disc_loss_type == 'vanilla':
            self.disc_loss = vanilla_d_loss
        elif disc_loss_type == 'hinge':
            self.disc_loss = hinge_d_loss
        else:
            raise NotImplementedError(
                f"Discriminator loss type {disc_loss_type} not implemented")

        self.perceptual_model = LPIPS().eval()

        self.image_gan_weight = image_gan_weight
        self.video_gan_weight = video_gan_weight

        self.perceptual_weight = perceptual_weight

        self.l1_weight = l1_weight

        self.segmentation_decoder = segmentation_decoder
        self.segmentation_criterion = segmentation_criterion
        self.classifier_head = classifier_head
        self.clasification_criterion = clasification_criterion
        self.bn = nn.BatchNorm3d(self.enc_out_ch)

        self.use_same_optimizer = use_same_optimizer

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        # self.dice = Dice(ignore_index=0)

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

    def encode(self, x, include_embeddings=False, quantize=True):
        h = self.pre_vq_conv(self.encoder(x))
        if quantize:
            vq_output = self.codebook(h)
            if include_embeddings:
                return vq_output['embeddings'], vq_output['encodings']
            else:
                return vq_output['encodings']
        return h

    def decode(self, latent, quantize=False):
        if quantize:
            vq_output = self.codebook(latent)
            latent = vq_output['encodings']
        h = F.embedding(latent, self.codebook.embeddings)
        h = self.post_vq_conv(shift_dim(h, -1, 1))
        return self.decoder(h)

    def forward_clasification(self, batch):
        x = batch['data']
        y = batch['label']
        logits = self.classifier_head(self.bn(self.encoder(x)))
        loss = self.clasification_criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def forward_segmentation(self, batch):
        x = batch['data']
        y = batch['mask'].long()
        label = batch['label']
        if isinstance(self.segmentation_criterion, (LossBinary, BCE_Lovasz)):
            cnt1 = (y == 1).sum().item()  # count number of class 1 in image
            cnt0 = y.numel() - cnt1
            if cnt1 != 0:
                BCE_pos_weight = torch.FloatTensor([1.0 * cnt0 / cnt1]).to(device=self.device)
            else:
                BCE_pos_weight = torch.FloatTensor([1.0]).to(device=self.device)
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

    def forward(self, x, optimizer_idx=None, log_image=False):
        B, C, T, H, W = x.shape

        z = self.pre_vq_conv(self.encoder(x))
        vq_output = self.codebook(z)
        x_recon = self.decoder(self.post_vq_conv(vq_output['embeddings']))

        recon_loss = F.l1_loss(x_recon, x) * self.l1_weight

        # Selects one random 2D image from each 3D Image
        frame_idx = torch.randint(0, T, [B]).cuda()
        frame_idx_selected = frame_idx.reshape(-1, 1, 1, 1, 1).repeat(1, C, 1, H, W)
        frames = torch.gather(x, 2, frame_idx_selected).squeeze(2)
        frames_recon = torch.gather(x_recon, 2, frame_idx_selected).squeeze(2)

        if log_image:
            return frames, frames_recon, x, x_recon

        if optimizer_idx == 0:
            # Autoencoder - train the "generator"

            # Perceptual loss
            perceptual_loss = 0
            if self.perceptual_weight > 0:
                perceptual_loss = self.perceptual_model(
                    frames, frames_recon).mean() * self.perceptual_weight

            # Discriminator loss (turned on after a certain epoch)
            logits_image_fake, pred_image_fake = self.image_discriminator(
                frames_recon)
            logits_video_fake, pred_video_fake = self.video_discriminator(
                x_recon)
            g_image_loss = -torch.mean(logits_image_fake)
            g_video_loss = -torch.mean(logits_video_fake)
            g_loss = self.image_gan_weight * g_image_loss + self.video_gan_weight * g_video_loss
            disc_factor = adopt_weight(
                self.global_step, threshold=self.discriminator_iter_start)
            aeloss = disc_factor * g_loss

            # GAN feature matching loss - tune features such that we get the same prediction result on the discriminator
            image_gan_feat_loss = 0
            video_gan_feat_loss = 0
            feat_weights = 4.0 / (3 + 1)
            if self.image_gan_weight > 0:
                logits_image_real, pred_image_real = self.image_discriminator(
                    frames)
                for i in range(len(pred_image_fake) - 1):
                    image_gan_feat_loss += feat_weights * \
                        F.l1_loss(pred_image_fake[i], pred_image_real[i].detach(
                        )) * (self.image_gan_weight > 0)
            if self.video_gan_weight > 0:
                logits_video_real, pred_video_real = self.video_discriminator(
                    x)
                for i in range(len(pred_video_fake) - 1):
                    video_gan_feat_loss += feat_weights * \
                        F.l1_loss(pred_video_fake[i], pred_video_real[i].detach(
                        )) * (self.video_gan_weight > 0)
            gan_feat_loss = disc_factor * self.gan_feat_weight * \
                (image_gan_feat_loss + video_gan_feat_loss)

            self.log("train/g_image_loss", g_image_loss,
                     logger=True, on_step=True, on_epoch=True)
            self.log("train/g_video_loss", g_video_loss,
                     logger=True, on_step=True, on_epoch=True)
            self.log("train/image_gan_feat_loss", image_gan_feat_loss,
                     logger=True, on_step=True, on_epoch=True)
            self.log("train/video_gan_feat_loss", video_gan_feat_loss,
                     logger=True, on_step=True, on_epoch=True)
            self.log("train/perceptual_loss", perceptual_loss,
                     prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("train/recon_loss", recon_loss, prog_bar=True,
                     logger=True, on_step=True, on_epoch=True)
            self.log("train/aeloss", aeloss, prog_bar=True,
                     logger=True, on_step=True, on_epoch=True)
            self.log("train/commitment_loss", vq_output['commitment_loss'],
                     prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log('train/perplexity', vq_output['perplexity'],
                     prog_bar=True, logger=True, on_step=True, on_epoch=True)
            return recon_loss, x_recon, vq_output, aeloss, perceptual_loss, gan_feat_loss

        if optimizer_idx == 1:
            # Train discriminator
            logits_image_real, _ = self.image_discriminator(frames.detach())
            logits_video_real, _ = self.video_discriminator(x.detach())

            logits_image_fake, _ = self.image_discriminator(
                frames_recon.detach())
            logits_video_fake, _ = self.video_discriminator(x_recon.detach())

            d_image_loss = self.disc_loss(logits_image_real, logits_image_fake)
            d_video_loss = self.disc_loss(logits_video_real, logits_video_fake)
            disc_factor = adopt_weight(
                self.global_step, threshold=self.discriminator_iter_start)
            discloss = disc_factor * \
                (self.image_gan_weight * d_image_loss + self.video_gan_weight * d_video_loss)

            self.log("train/logits_image_real", logits_image_real.mean().detach(),
                     logger=True, on_step=True, on_epoch=True)
            self.log("train/logits_image_fake", logits_image_fake.mean().detach(),
                     logger=True, on_step=True, on_epoch=True)
            self.log("train/logits_video_real", logits_video_real.mean().detach(),
                     logger=True, on_step=True, on_epoch=True)
            self.log("train/logits_video_fake", logits_video_fake.mean().detach(),
                     logger=True, on_step=True, on_epoch=True)
            self.log("train/d_image_loss", d_image_loss,
                     logger=True, on_step=True, on_epoch=True)
            self.log("train/d_video_loss", d_video_loss,
                     logger=True, on_step=True, on_epoch=True)
            self.log("train/discloss", discloss, prog_bar=True,
                     logger=True, on_step=True, on_epoch=True)
            return discloss

        perceptual_loss = self.perceptual_model(
            frames, frames_recon) * self.perceptual_weight
        return recon_loss, x_recon, vq_output, perceptual_loss

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def training_step(self, batch, batch_idx):
        x = batch['data']
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

        # Train the autoencoder
        opt_ae.zero_grad()
        recon_loss, _, vq_output, aeloss, perceptual_loss, gan_feat_loss = self.forward(
            x, optimizer_idx=0)
        commitment_loss = vq_output['commitment_loss']
        loss = recon_loss + commitment_loss + aeloss + perceptual_loss + gan_feat_loss
        self.manual_backward(loss)
        opt_ae.step()

        # Train the discriminator
        opt_disc.zero_grad()
        discloss = self.forward(x, optimizer_idx=1)
        self.manual_backward(discloss)
        opt_disc.step()
        if self.use_same_optimizer:
            opt_ds.zero_grad()
            seg_loss, seg_preds, seg_targets = self.forward_segmentation(batch)
            cls_loss, cls_preds, cls_targets = self.forward_clasification(batch)
            ds_loss = seg_loss + cls_loss
            self.manual_backward(ds_loss)
            opt_ds.step()
            self.log("train/ds_loss", ds_loss, prog_bar=True,
                     logger=True, on_step=True, on_epoch=True)
        else:
            if self.segmentation_decoder is not None:
                opt_seg.zero_grad()
                seg_loss, seg_preds, seg_targets = self.forward_segmentation(batch)
                self.manual_backward(seg_loss)
                opt_seg.step()
            if self.classifier_head is not None:
                opt_cls.zero_grad()
                cls_loss, cls_preds, cls_targets = self.forward_clasification(batch)
                self.manual_backward(cls_loss)
                opt_cls.step()

        if self.classifier_head is not None and self.segmentation_decoder is not None:
            return {"seg_loss": seg_loss, "seg_preds": seg_preds, "seg_targets": seg_targets,
                    "cls_loss": cls_loss, "cls_preds": cls_preds, "cls_targets": cls_targets}
        elif self.classifier_head is None and self.segmentation_decoder is not None:
            return {"seg_loss": seg_loss, "seg_preds": seg_preds, "seg_targets": seg_targets}

    # def on_validation_epoch_end(self):
    #     self.dice.reset()

    def validation_step(self, batch, batch_idx):
        x = batch['data']  # TODO: batch['stft']
        with self.ema_scope():
            recon_loss, _, vq_output, perceptual_loss = self.forward(x)
        if self.segmentation_decoder is not None:
            seg_loss, seg_preds, seg_targets = self.forward_segmentation(batch)
        if self.classifier_head is not None:
            cls_loss, cls_preds, cls_targets = self.forward_clasification(batch)

        # B, C, T, W, H = seg_preds.shape
        # self.dice(seg_preds.view(B, -1).to(self.device), seg_targets.to(self.device).view(B, -1))
        # self.log('val/module_dice', self.dice.compute(), on_step=False, on_epoch=True, prog_bar=False)

        self.log('val/recon_loss', recon_loss, prog_bar=True)
        self.log('val/perceptual_loss', perceptual_loss, prog_bar=True)
        self.log('val/perplexity', vq_output['perplexity'], prog_bar=True)
        self.log('val/commitment_loss', vq_output['commitment_loss'], prog_bar=True)
        if self.segmentation_decoder is not None and self.classifier_head is not None:
            return {"seg_loss": seg_loss, "seg_preds": seg_preds, "seg_targets": seg_targets,
                    "cls_loss": cls_loss, "cls_preds": cls_preds, "cls_targets": cls_targets}
        elif self.classifier_head is None and self.segmentation_decoder is not None:
            return {"seg_loss": seg_loss, "seg_preds": seg_preds, "seg_targets": seg_targets}

    def test_step(self, batch, batch_idx):
        x = batch['data']  # TODO: batch['stft']

        if self.use_ema:
            with self.ema_scope():
                recon_loss, _, vq_output, perceptual_loss = self.forward(x)
                if self.segmentation_decoder is not None:
                    seg_loss, seg_preds, seg_targets = self.forward_segmentation(batch)
                if self.classifier_head is not None:
                    cls_loss, cls_preds, cls_targets = self.forward_clasification(batch)
        else:
            recon_loss, _, vq_output, perceptual_loss = self.forward(x)
            if self.segmentation_decoder is not None:
                seg_loss, seg_preds, seg_targets = self.forward_segmentation(batch)
            if self.classifier_head is not None:
                cls_loss, cls_preds, cls_targets = self.forward_clasification(batch)

        self.log('test/recon_loss', recon_loss, prog_bar=True)
        self.log('test/perceptual_loss', perceptual_loss, prog_bar=True)
        self.log('test/perplexity', vq_output['perplexity'], prog_bar=True)
        self.log('test/commitment_loss',
                 vq_output['commitment_loss'], prog_bar=True)
        if self.segmentation_decoder is not None and self.classifier_head is not None:
            return {"seg_loss": seg_loss, "seg_preds": seg_preds, "seg_targets": seg_targets,
                    "cls_loss": cls_loss, "cls_preds": cls_preds, "cls_targets": cls_targets}
        elif self.classifier_head is None and self.segmentation_decoder is not None:
            return {"seg_loss": seg_loss, "seg_preds": seg_preds, "seg_targets": seg_targets}

    def configure_optimizers(self):
        lr = self.lr
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) + \
                                  list(self.decoder.parameters()) + \
                                  list(self.pre_vq_conv.parameters()) + \
                                  list(self.post_vq_conv.parameters()) + \
                                  list(self.codebook.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(list(self.image_discriminator.parameters()) + \
                                    list(self.video_discriminator.parameters()),
                                    lr=lr, betas=(0.5, 0.9))
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
                                        list(self.encoder.parameters()), lr=0.01, betas=(0.5, 0.9))
                opt_list.append(opt_seg)
            if self.classifier_head is not None:
                opt_cls = torch.optim.Adam(list(self.classifier_head.parameters()) + \
                                        list(self.encoder.parameters()), lr=lr, betas=(0.5, 0.9))
                opt_list.append(opt_cls)

        return (opt_list, [])

    def log_images(self, batch, **kwargs):
        log = dict()
        x = batch['data']
        x = x.to(self.device)

        if self.use_ema:
            with self.ema_scope():
                frames, frames_rec, _, _ = self(x, log_image=True)
        else:
            frames, frames_rec, _, _ = self(x, log_image=True)

        log["inputs"] = frames
        log["reconstructions"] = frames_rec
        # log['mean_org'] = batch['mean_org']
        # log['std_org'] = batch['std_org']
        return log

    def log_videos(self, batch, **kwargs):
        log = dict()
        x = batch['data']

        if self.use_ema:
            with self.ema_scope():
                _, _, x, x_rec = self(x, log_image=True)
        else:
            _, _, x, x_rec = self(x, log_image=True)

        log["inputs"] = x
        log["reconstructions"] = x_rec
        # log['mean_org'] = batch['mean_org']
        # log['std_org'] = batch['std_org']
        return log


@hydra.main(
    version_base="1.3", config_path="../../configs/model", config_name="multihead_vq_gan_3d.yaml"
)
def main(cfg: DictConfig):
    import shutil
    # shutil.rmtree('outputs')
    print(cfg)
    # cfg.embedding_dim = 16
    # cfg.n_hiddens = 16
    # cfg.downsample = [4, 4, 4]
    # cfg.disc_layers = 1
    model: MultiheadVQGAN = hydra.utils.instantiate(cfg).to('cuda')
    input_tensor = torch.randn(1, 1, 128, 128, 128).to('cuda')
    encoded_output = model.encoder(input_tensor)
    from IPython import embed
    embed()
    print(encoded_output.shape)
    print('Encoder out channels:', model.encoder.out_channels)

if __name__ == "__main__":
    main()
