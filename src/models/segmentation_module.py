import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from contextlib import contextmanager
from typing import Any, List
import gc
import hydra
import torch
from omegaconf import DictConfig
from lightning import LightningModule
from torchmetrics import Dice, JaccardIndex, MaxMetric, MeanMetric

# from src.models import load_autoencoder
from src.models.components.loss_function.lossbinary import LossBinary
from src.models.components.loss_function.lovasz_loss import BCE_Lovasz
import torch.nn.functional as F
from src.utils.ema import LitEma

class SegmentationModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        criterion: torch.nn.Module,
        use_ema: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net", "criterion"])

        self.net = net

        # loss function
        self.criterion = criterion
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.net)

    def forward(self, x: torch.Tensor):
        return self.net(x)
    
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

    def model_step(self, batch: Any):
        x = batch['segmentation']
        y = batch['mask'].long()
        if isinstance(self.criterion, (LossBinary, BCE_Lovasz)):
            cnt1 = (y == 1).sum().item()  # count number of class 1 in image
            cnt0 = y.numel() - cnt1
            if cnt1 != 0:
                BCE_pos_weight = torch.FloatTensor([1.0 * cnt0 / cnt1]).to(device=self.device)
            else:
                BCE_pos_weight = torch.FloatTensor([1.0]).to(device=self.device)

            self.criterion.update_pos_weight(pos_weight=BCE_pos_weight)

        # from IPython import embed; embed()
        logits = self.forward(x)
        if self.net.n_classes == 2:
            loss = self.criterion(logits, y.squeeze(1))
            preds = torch.argmax(logits, dim=1).unsqueeze(0)
            preds = preds.permute(1, 0, 2, 3)
        else:
            loss = self.criterion(logits, y.float())
            preds = torch.sigmoid(logits)
            preds[preds >= 0.5] = 1
            preds[preds < 0.5] = 0
        # from IPython import embed; embed()
        # Code to try to fix CUDA out of memory issues
        del x
        gc.collect()
        torch.cuda.empty_cache()

        return loss, preds, y

    def forward_segmentation(self, batch):
        return self.model_step(batch)

    def training_step(self, batch: Any, batch_idx: int):
        seg_loss, seg_preds, seg_targets = self.model_step(batch)
        return {"seg_loss": seg_loss, "seg_preds": seg_preds, "seg_targets": seg_targets, "loss": seg_loss}

    def validation_step(self, batch: Any, batch_idx: int):
        with self.ema_scope():
            seg_loss, seg_preds, seg_targets = self.model_step(batch)
        return {"seg_loss": seg_loss, "seg_preds": seg_preds, "seg_targets": seg_targets, "loss": seg_loss}

    def test_step(self, batch: Any, batch_idx: int):
        with self.ema_scope():
            seg_loss, seg_preds, seg_targets = self.model_step(batch)
        return {"seg_loss": seg_loss, "seg_preds": seg_preds, "seg_targets": seg_targets, "loss": seg_loss}

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/dice",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


@hydra.main(
    version_base="1.3", config_path="../../configs/model", config_name="downstream_segmentation.yaml"
)
def main(cfg: DictConfig):
    # import shutil
    # shutil.rmtree('outputs')
    print(cfg)
    cfg.net.autoencoder_path = "./outputs/vq_gan_3d_low_compression/lung-thesis/2aglgm52/checkpoints/epoch=111-step=179200.ckpt"
    from IPython import embed
    embed()
    model = hydra.utils.instantiate(cfg)
    input_tensor = torch.randn(1, 1, 128, 128, 128) # .to('cuda')
    output = model(input_tensor)
    print(output.shape)
    # encoded_output = model.encoder(input_tensor)
    # print(encoded_output.shape)
    # print('Encoder out channels:', model.encoder.out_channels)

if __name__ == "__main__":
    main()
