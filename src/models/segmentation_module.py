import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


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
        loss_weight = None
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net", "criterion"])

        self.net = net

        # loss function
        self.criterion = criterion
        if loss_weight is not None:
            self.criterion = torch.nn.CrossEntropyLoss(weight=torch.Tensor(loss_weight))
            # self.criterion = torch.nn.CrossEntropyLoss(weight=torch.Tensor(loss_weight), ignore_index=0)

        # metric objects for calculating and averaging accuracy across batches
        self.train_jaccard = JaccardIndex(task="binary", num_classes=2)
        self.val_jaccard = JaccardIndex(task="binary", num_classes=2)
        self.test_jaccard = JaccardIndex(task="binary", num_classes=2)

        self.train_dice = Dice(num_classes=2, ignore_index=0)
        self.val_dice = Dice(num_classes=2, ignore_index=0)
        self.test_dice = Dice(num_classes=2, ignore_index=0)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_metric_best_1 = MaxMetric()
        self.val_metric_best_2 = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_jaccard.reset()
        self.val_dice.reset()
        self.val_metric_best_1.reset()
        self.val_metric_best_2.reset()

    def model_step(self, batch: Any):
        x = batch['data']
        y = batch['mask'].long()
        # from IPython import embed; embed()
        label = batch['label']
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
        loss = self.criterion(logits, y.squeeze(1))
        preds = torch.argmax(logits, dim=1).unsqueeze(0)
        # from IPython import embed; embed()
        # Code to try to fix CUDA out of memory issues
        del x
        gc.collect()
        torch.cuda.empty_cache()

        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_jaccard(preds, targets)
        self.train_dice(preds, targets.int())

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/jaccard", self.train_jaccard, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/dice", self.train_dice, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_jaccard(preds, targets)
        self.val_dice(preds, targets.int())

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val/jaccard",
            self.val_jaccard,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/dice",
            self.val_dice,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return {"loss": loss, "preds": preds, "targets": targets}

    def on_validation_epoch_end(self):
        # get current val acc
        acc1 = self.val_jaccard.compute()
        acc2 = self.val_dice.compute()
        # update best so far val acc
        self.val_metric_best_1(acc1)
        self.val_metric_best_2(acc2)
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/jaccard_best", self.val_metric_best_1.compute(), prog_bar=True)
        self.log("val/dice_best", self.val_metric_best_2.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        # update and log metrics
        self.test_loss(loss)
        self.test_jaccard(preds, targets)
        self.test_dice(preds, targets.int())

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/jaccard", self.test_jaccard, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/dice", self.test_dice, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

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
                    "monitor": "val/loss",
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
    from IPython import embed; embed()
    model = hydra.utils.instantiate(cfg)
    input_tensor = torch.randn(1, 1, 128, 128, 128) #.to('cuda')
    output = model(input_tensor)
    print(output.shape)
    # encoded_output = model.encoder(input_tensor)
    # print(encoded_output.shape)
    # print('Encoder out channels:', model.encoder.out_channels)

if __name__ == "__main__":
    main()