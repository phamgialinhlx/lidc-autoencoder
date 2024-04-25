import math
import torch
from lightning.pytorch.callbacks import Callback
from torchmetrics import Dice, JaccardIndex, MaxMetric, MeanMetric
from torchvision.utils import make_grid
import wandb

class Mask2DLogger(Callback):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.last_batch = None
        self.last_output = None

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.last_batch = batch
        self.last_output = outputs

    def log_img(self, trainer, outputs, batch):
        preds = outputs["seg_preds"].float()
        preds = preds.permute(1, 0, 2, 3)

        nrows = math.ceil(math.sqrt(preds.shape[0]))
        value_range = (0, 1)
        seg_img = make_grid(
            batch["segmentation"], nrow=nrows, normalize=False, value_range=value_range
        )
        targets = make_grid(
            batch["mask"], nrow=nrows, normalize=False, value_range=value_range
        )
        preds = make_grid(
            preds, nrow=nrows, normalize=False, value_range=value_range
        )
        trainer.logger.experiment.log(
            {
                "image": [
                    wandb.Image(seg_img),
                    wandb.Image(targets),
                    wandb.Image(preds),
                ],
                "caption": ["segmentation", "targets", "preds"],
            }
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        # if not trainer.sanity_checking:
        self.log_img(trainer, self.last_output, self.last_batch)
