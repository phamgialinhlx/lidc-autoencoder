import torch
from lightning.pytorch.callbacks import Callback
from torchmetrics import Dice, JaccardIndex, MaxMetric, MeanMetric

class SegmentationMetrics(Callback):
    def __init__(self):
        super().__init__(device="cpu")
        self.train_jaccard = JaccardIndex(task="binary", num_classes=2)
        self.val_jaccard = JaccardIndex(task="binary", num_classes=2)
        self.test_jaccard = JaccardIndex(task="binary", num_classes=2)

        self.train_dice = Dice(num_classes=2, ignore_index=0)
        self.val_dice = Dice(num_classes=2, ignore_index=0)
        self.test_dice = Dice(num_classes=2, ignore_index=0)

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        
        # self.val_metric_best_1 = MaxMetric()
        # self.val_metric_best_2 = MaxMetric()
        self.val_metric_best_1 = float("-inf")
        self.val_metric_best_2 = float("-inf")

    def on_fit_start(self, trainer, pl_module):
        self.train_jaccard = self.train_jaccard.to(self.device)
        self.val_jaccard = self.val_jaccard.to(self.device)
        self.test_jaccard = self.test_jaccard.to(self.device)
        
        self.train_dice = self.train_dice.to(self.device)
        self.val_dice = self.val_dice.to(self.device)
        self.test_dice = self.test_dice.to(self.device)
        
        self.train_loss = self.train_loss.to(self.device)
        self.val_loss = self.val_loss.to(self.device)
        self.test_loss = self.test_loss.to(self.device)

    # def on_train_start(self):
    #     # by default lightning executes validation step sanity checks before training starts,
    #     # so it's worth to make sure validation metrics don't store results from these checks
    #     # self.val_loss.reset()
    #     self.val_jaccard.reset()
    #     self.val_dice.reset()
        # self.val_metric_best_1.reset()
        # self.val_metric_best_2.reset()
            
    def on_train_epoch_start(self, trainer, pl_module):
        self.train_jaccard.reset()
        self.train_dice.reset()
        self.train_loss.reset()

    def on_validation_epoch_start(self, trainer, pl_module):
        self.val_jaccard.reset()
        self.val_dice.reset()
        self.val_loss.reset()

    def on_test_epoch_start(self, trainer, pl_module):
        self.test_jaccard.reset()
        self.test_dice.reset()
        self.test_loss.reset()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs["seg_loss"]
        preds = outputs["seg_preds"]
        targets = batch["mask"].long()

        self.train_loss(loss.to(self.device))
        self.train_jaccard(preds.to(self.device), targets.to(self.device))
        self.train_dice(preds.to(self.device), targets.int().to(self.device))

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        loss = outputs["seg_loss"]
        preds = outputs["seg_preds"]
        targets = batch["mask"].long()

        self.val_loss(loss.to(self.device))
        self.val_jaccard(preds.to(self.device), targets.to(self.device))
        self.val_dice(preds.to(self.device), targets.int().to(self.device))

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        loss = outputs["seg_loss"]
        preds = outputs["seg_preds"]
        targets = batch["mask"].long()

        self.test_loss(loss.to(self.device))
        self.test_jaccard(preds.to(self.device), targets.to(self.device))
        self.test_dice(preds.to(self.device), targets.int().to(self.device))

    def on_validation_epoch_end(self, trainer, pl_module):
        pl_module.log("train/seg_loss", self.train_loss, metric_attribute="train_loss", on_step=False, on_epoch=True, prog_bar=True)
        pl_module.log("train/jaccard", self.train_jaccard.compute())
        pl_module.log("train/dice", self.train_dice.compute())

        val_jaccard = self.val_jaccard.compute()
        val_dice = self.val_dice.compute()
        pl_module.log("val/seg_loss", self.val_loss, metric_attribute="val_loss", on_step=False, on_epoch=True, prog_bar=True)
        pl_module.log("val/jaccard", val_jaccard)
        pl_module.log("val/dice", val_dice)

        self.val_metric_best_1 = max(self.val_metric_best_1, val_jaccard)
        self.val_metric_best_2 = max(self.val_metric_best_2, val_dice)
        pl_module.log("val/jaccard_best", self.val_metric_best_1)
        pl_module.log("val/dice_best", self.val_metric_best_2)

        if trainer.testing:
            pl_module.log("test/seg_loss", self.test_loss, metric_attribute="test_loss", on_step=False, on_epoch=True, prog_bar=True)
            pl_module.log("test/jaccard", self.test_jaccard.compute())
            pl_module.log("test/dice", self.test_dice.compute())
    # def on_validation_epoch_end(self):
    #     # get current val acc
    #     acc1 = self.val_jaccard.compute()
    #     acc2 = self.val_dice.compute()
    #     # update best so far val acc
    #     self.val_metric_best_1(acc1)
    #     self.val_metric_best_2(acc2)
    #     # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
    #     # otherwise metric would be reset by lightning after each epoch
    #     # self.log("val/jaccard_best", self.val_metric_best_1.compute(), prog_bar=True)
    #     # self.log("val/dice_best", self.val_metric_best_2.compute(), prog_bar=True)
