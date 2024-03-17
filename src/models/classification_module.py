from typing import Any, Dict, Tuple

import hydra
import torch
from omegaconf import DictConfig
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall, ROC, CohenKappa, AUC

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.vq_gan_3d_module import VQGAN
from src.models.diffusion_module import load_autoencoder

class ClassificationModule(LightningModule):
    def __init__(self, 
                net, 
                autoencoder_ckpt_path: str,
                optimizer: torch.optim.Optimizer, 
                scheduler: torch.optim.lr_scheduler, 
                compile: bool, 
                num_classes: int = 2,
                *args: Any, 
                **kwargs: Any
        ) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(logger=False)
        
        self.autoencoder = load_autoencoder(autoencoder_ckpt_path, disable_decoder=True)
        self.net = net        
        
        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()
        
        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

        self.train_f1 = F1Score(num_classes=num_classes)
        self.train_precision = Precision(num_classes=num_classes)
        self.train_recall = Recall(num_classes=num_classes)
        self.train_kappa = CohenKappa(num_classes=num_classes)
        self.train_auc = AUC(num_classes=num_classes)

        self.val_f1 = F1Score(num_classes=num_classes)
        self.val_precision = Precision(num_classes=num_classes)
        self.val_recall = Recall(num_classes=num_classes)
        self.val_kappa = CohenKappa(num_classes=num_classes)
        self.val_auc = AUC(num_classes=num_classes)

        self.test_f1 = F1Score(num_classes=num_classes)
        self.test_precision = Precision(num_classes=num_classes)
        self.test_recall = Recall(num_classes=num_classes)
        self.test_kappa = CohenKappa(num_classes=num_classes)
        self.test_auc = AUC(num_classes=num_classes)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.autoencoder, VQGAN):
                x = self.autoencoder.encode(
                    x, quantize=False, include_embeddings=True)
        else:
            x = self.autoencoder.encode(x).sample()
        return self.net(x)
    
    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = batch['data']
        y = batch['label']
        
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, preds, y = self.model_step(batch)
        self.train_loss(loss)
        self.train_acc(preds, y)
        self.train_f1(preds, y)
        self.train_precision(preds, y)
        self.train_recall(preds, y)
        self.train_kappa(preds, y)
        self.train_auc(preds, y)

        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/cohen_kappa", self.train_kappa, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/precision", self.train_precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/auc", self.train_auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/recall", self.train_recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.val_f1(preds, targets)
        self.val_precision(preds, targets)
        self.val_recall(preds, targets)
        self.val_kappa(preds, targets)
        self.val_auc(preds, targets)

        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/cohen_kappa", self.val_kappa, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/precision", self.val_precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auc", self.val_auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/recall", self.val_recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.test_f1(preds, targets)
        self.test_precision(preds, targets)
        self.test_recall(preds, targets)
        self.test_kappa(preds, targets)
        self.test_auc(preds, targets)

        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/cohen_kappa", self.test_kappa, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/precision", self.test_precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/auc", self.test_auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/recall", self.test_recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

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
                    "monitor": "val/loss",
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