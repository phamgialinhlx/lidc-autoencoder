import torch
import math
import wandb
from torchvision.utils import make_grid
import lightning as pl
from lightning.pytorch.callbacks.callback import Callback
from torchvision.transforms import transforms
from IPython import embed

class ImageLogger(Callback):
    def __init__(self, frequency: int = 1, seg_head: bool = False, input_key="segmentation"):
        super().__init__()
        self.count = 0
        self.frequency = frequency
        self.seg_head = seg_head
        self.input_key = input_key

    @torch.no_grad()
    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.count += 1
        if self.count % self.frequency == 0:
            origin = next(iter(trainer.val_dataloaders))['segmentation'].to(pl_module.device)

            image = pl_module.log_image(origin, device=pl_module.device)

            nrows = math.ceil(math.sqrt(image.shape[0]))

            value_range = (0, 1)

            origin = make_grid(
                origin, nrow=nrows, normalize=True, value_range=value_range
            )
            image = make_grid(
                image, nrow=nrows, normalize=True, value_range=value_range
            )

            trainer.logger.experiment.log(
                {
                    "image": [
                        wandb.Image(origin),
                        wandb.Image(image),
                    ],
                    "caption": ["origin", "generate"],
                }
            )