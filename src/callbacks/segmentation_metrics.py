import torch
from torchmetrics import Dice, JaccardIndex
from lightning.pytorch.callbacks import Callback

class SegmentationMetrics(Callback):
    def __init__(self):
        super().__init__()
        self.train_jaccard = JaccardIndex(task="binary", num_classes=2)
        self.val_jaccard = JaccardIndex(task="binary", num_classes=2)
        self.test_jaccard = JaccardIndex(task="binary", num_classes=2)

        self.train_dice = Dice(num_classes=2, ignore_index=0)
        self.val_dice = Dice(num_classes=2, ignore_index=0)
        self.test_dice = Dice(num_classes=2, ignore_index=0)

        self.train_loss = torch.tensor(0.0)
        self.val_loss = torch.tensor(0.0)
        self.test_loss = torch.tensor(0.0)

        self.val_metric_best_1 = float("-inf")
        self.val_metric_best_2 = float("-inf")

    def on_train_epoch_start(self, trainer, pl_module):
        self.train_jaccard.reset()
        self.train_dice.reset()
        self.train_loss = 0.0

    def on_validation_epoch_start(self, trainer, pl_module):
        self.val_jaccard.reset()
        self.val_dice.reset()
        self.val_loss = 0.0

    def on_test_epoch_start(self, trainer, pl_module):
        self.test_jaccard.reset()
        self.test_dice.reset()
        self.test_loss = 0.0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs["loss"]
        preds = outputs["preds"]
        targets = batch["mask"].long()

        self.train_loss += loss.detach()
        self.train_jaccard(preds, targets)
        self.train_dice(preds, targets.int())

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        loss = outputs["loss"]
        preds = outputs["preds"]
        targets = batch["mask"].long()

        self.val_loss += loss.detach()
        self.val_jaccard(preds, targets)
        self.val_dice(preds, targets.int())

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        loss = outputs["loss"]
        preds = outputs["preds"]
        targets = batch["mask"].long()

        self.test_loss += loss.detach()
        self.test_jaccard(preds, targets)
        self.test_dice(preds, targets.int())

    def on_epoch_end(self, trainer, pl_module):
        pl_module.log("train/loss", self.train_loss / len(trainer.train_dataloader))
        pl_module.log("train/jaccard", self.train_jaccard.compute())
        pl_module.log("train/dice", self.train_dice.compute())

        val_jaccard = self.val_jaccard.compute()
        val_dice = self.val_dice.compute()
        pl_module.log("val/loss", self.val_loss / len(trainer.val_dataloaders[0]))
        pl_module.log("val/jaccard", val_jaccard)
        pl_module.log("val/dice", val_dice)

        self.val_metric_best_1 = max(self.val_metric_best_1, val_jaccard)
        self.val_metric_best_2 = max(self.val_metric_best_2, val_dice)
        pl_module.log("val/jaccard_best", self.val_metric_best_1)
        pl_module.log("val/dice_best", self.val_metric_best_2)

        if trainer.testing:
            pl_module.log("test/loss", self.test_loss / len(trainer.test_dataloaders[0]))
            pl_module.log("test/jaccard", self.test_jaccard.compute())
            pl_module.log("test/dice", self.test_dice.compute())
