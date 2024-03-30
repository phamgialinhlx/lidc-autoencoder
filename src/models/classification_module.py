from typing import Any, Dict, List, Tuple
from contextlib import contextmanager

import hydra
import torch
from omegaconf import DictConfig
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall, ROC, CohenKappa, AUC

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.ema import LitEma
from src.models.vq_gan_3d_module import VQGAN
from src.models.diffusion_module import load_autoencoder
import torch.nn.functional as F

class ClassificationModule(LightningModule):
    def __init__(self, 
                net, 
                criterion,
                autoencoder_ckpt_path: str,
                optimizer: torch.optim.Optimizer, 
                scheduler: torch.optim.lr_scheduler, 
                compile: bool, 
                num_classes: int = 2,
                use_ema: bool = True,
                *args: Any, 
                **kwargs: Any
        ) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(logger=False)
        
        self.autoencoder = load_autoencoder(autoencoder_ckpt_path, disable_decoder=True, eval=False)
        self.net = net        
        
        # loss function
        self.criterion = criterion 

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.autoencoder, VQGAN):
                # x = self.autoencoder.encode(
                    # x, quantize=False, include_embeddings=True)
            x = self.autoencoder.encoder(x)
        else:
            x = self.autoencoder.encode(x).sample()
        return self.net(x)

    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = batch['data']
        y = batch['label']
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, preds, y = self.model_step(batch)
        return {"cls_loss": loss, "cls_preds": preds, "cls_targets": y, "loss": loss}
        
    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        if self.use_ema:
            with self.ema_scope():
                loss, preds, targets = self.model_step(batch)
        else:
            loss, preds, targets = self.model_step(batch)
        
        return {"cls_loss": loss, "cls_preds": preds, "cls_targets": targets}

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)
        if self.use_ema:
            with self.ema_scope():
                loss, preds, targets = self.model_step(batch)
        else:
            loss, preds, targets = self.model_step(batch)
        
        return {"cls_loss": loss, "cls_preds": preds, "cls_targets": targets}

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/cls_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

@hydra.main(
    version_base="1.3", config_path="../../configs/model", config_name="downstream_classification.yaml"
)
def main(cfg: DictConfig):
    # import shutil
    # shutil.rmtree('outputs')
    print(cfg)
    cfg.autoencoder_ckpt_path = "./outputs/vq_gan_3d_low_compression/lung-thesis/2aglgm52/checkpoints/epoch=111-step=179200.ckpt"
    from IPython import embed; embed()
    model = hydra.utils.instantiate(cfg)
    model.autoencoder = load_autoencoder(cfg.autoencoder_ckpt_path, disable_decoder=True, map_location='cpu')
    input_tensor = torch.randn(1, 1, 128, 128, 128) #.to('cuda')
    output = model(input_tensor)
    print(output.shape)
    # encoded_output = model.encoder(input_tensor)
    # print(encoded_output.shape)
    # print('Encoder out channels:', model.encoder.out_channels)

if __name__ == "__main__":
    main()