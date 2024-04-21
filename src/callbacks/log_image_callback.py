import torch
import math
import wandb
from torchvision.utils import make_grid
import lightning as pl
from lightning.pytorch.callbacks.callback import Callback
from torchvision.transforms import transforms
from IPython import embed

class LogImageCallback(Callback):
    def __init__(self, frequency: int = 1, seg_head: bool = False):
        super().__init__()
        self.count = 0
        self.frequency = frequency
        self.seg_head = seg_head

    @torch.no_grad()
    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.count += 1
        if self.count % self.frequency == 0:
            if self.seg_head:
                origin = next(iter(trainer.val_dataloaders))['data'].to(pl_module.device)
                image = pl_module.log_image(origin, device=pl_module.device)
                seg_image = next(iter(trainer.val_dataloaders))['segmentation'].to(pl_module.device)
                nrows = math.ceil(math.sqrt(image.shape[0]))

                value_range = (-1, 1)
                compare = make_grid(
                    torch.cat([seg_image, image], dim=3),
                    nrow=nrows,
                    normalize=True,
                    value_range=value_range,
                )

                origin = make_grid(
                    origin, nrow=nrows, normalize=True, value_range=value_range
                )
                seg_img = make_grid(
                    seg_image, nrow=nrows, normalize=True, value_range=value_range
                )
                image = make_grid(
                    image, nrow=nrows, normalize=True, value_range=value_range
                )

                trainer.logger.experiment.log(
                    {
                        "image": [
                            wandb.Image(origin),
                            wandb.Image(seg_img),
                            wandb.Image(image),
                            wandb.Image(compare),
                        ],
                        "caption": ["origin", "reconstruct", "compare"],
                    }
                )
            else:
                origin = next(iter(trainer.val_dataloaders))['data'].to(pl_module.device)
                
                image = pl_module.log_image(origin, device=pl_module.device)

                nrows = math.ceil(math.sqrt(image.shape[0]))

                value_range = (-1, 1)
                compare = make_grid(
                    torch.cat([origin, image], dim=3),
                    nrow=nrows,
                    normalize=True,
                    value_range=value_range,
                )

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
                            wandb.Image(compare),
                        ],
                        "caption": ["origin", "reconstruct", "compare"],
                    }
                )
