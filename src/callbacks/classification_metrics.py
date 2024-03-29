import torch
from torchmetrics import Accuracy, F1Score, Precision, Recall, CohenKappa
from torchmetrics import MaxMetric, MeanMetric
from lightning.pytorch.callbacks import Callback

class ClassificationMetrics(Callback):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes, average="micro")
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes, average="micro")
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes, average="micro")

        self.train_f1 = F1Score(num_classes=num_classes, average="micro")
        self.train_precision = Precision(num_classes=num_classes, average="micro")
        self.train_recall = Recall(num_classes=num_classes, average="micro")
        self.train_kappa = CohenKappa(num_classes=num_classes)

        self.val_f1 = F1Score(num_classes=num_classes, average="micro")
        self.val_precision = Precision(num_classes=num_classes, average="micro")
        self.val_recall = Recall(num_classes=num_classes, average="micro")
        self.val_kappa = CohenKappa(num_classes=num_classes)

        self.test_f1 = F1Score(num_classes=num_classes, average="micro")
        self.test_precision = Precision(num_classes=num_classes, average="micro")
        self.test_recall = Recall(num_classes=num_classes, average="micro")
        self.test_kappa = CohenKappa(num_classes=num_classes)

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_acc_best = float("-inf")
        
    def on_fit_start(self, trainer, pl_module):
        self.train_acc = self.train_acc.to(pl_module.device)
        self.val_acc = self.val_acc.to(pl_module.device)
        self.test_acc = self.test_acc.to(pl_module.device)

        self.train_f1 = self.train_f1.to(pl_module.device)
        self.train_precision = self.train_precision.to(pl_module.device)
        self.train_recall = self.train_recall.to(pl_module.device)
        self.train_kappa = self.train_kappa.to(pl_module.device)

        self.val_f1 = self.val_f1.to(pl_module.device)
        self.val_precision = self.val_precision.to(pl_module.device)
        self.val_recall = self.val_recall.to(pl_module.device)
        self.val_kappa = self.val_kappa.to(pl_module.device)

        self.test_f1 = self.test_f1.to(pl_module.device)
        self.test_precision = self.test_precision.to(pl_module.device)
        self.test_recall = self.test_recall.to(pl_module.device)
        self.test_kappa = self.test_kappa.to(pl_module.device)
        
        self.train_loss = self.train_loss.to(pl_module.device)
        self.val_loss = self.val_loss.to(pl_module.device)
        self.test_loss = self.test_loss.to(pl_module.device)

    def reset(self):
    # def on_train_epoch_start(self, trainer, pl_module):
        self.train_acc.reset()
        self.train_f1.reset()
        self.train_precision.reset()
        self.train_recall.reset()
        self.train_kappa.reset()
        self.train_loss.reset()
    # def on_validation_epoch_start(self, trainer, pl_module):
        self.val_acc.reset()
        self.val_f1.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_kappa.reset()
        self.val_loss.reset()


    # def on_test_epoch_start(self, trainer, pl_module):
        self.test_acc.reset()
        self.test_f1.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_kappa.reset()
        self.test_loss.reset()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs["cls_loss"]
        preds = outputs["cls_preds"]
        targets = batch["label"]

        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.train_f1(preds, targets)
        self.train_precision(preds, targets)
        self.train_recall(preds, targets)
        self.train_kappa(preds, targets)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        loss = outputs["cls_loss"]
        preds = outputs["cls_preds"]
        targets = batch["label"]

        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.val_f1(preds, targets)
        self.val_precision(preds, targets)
        self.val_recall(preds, targets)
        self.val_kappa(preds, targets)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        loss = outputs["cls_loss"]
        preds = outputs["cls_preds"]
        targets = batch["label"]

        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.test_f1(preds, targets)
        self.test_precision(preds, targets)
        self.test_recall(preds, targets)
        self.test_kappa(preds, targets)

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.sanity_checking:
            self.train_acc.compute()
            self.train_f1.compute()
            self.train_precision.compute()
            self.train_recall.compute()
            self.train_kappa.compute()
            self.val_acc.compute()
            self.val_f1.compute()
            self.val_precision.compute()
            self.val_recall.compute()
            self.val_kappa.compute()
            
            
            pl_module.log("train/cls_loss", self.train_loss, metric_attribute="train_loss", on_step=False, on_epoch=True, prog_bar=True)
            pl_module.log("train/acc", self.train_acc, metric_attribute="train_acc", on_step=False, on_epoch=True, prog_bar=True)
            pl_module.log("train/f1", self.train_f1, metric_attribute="train_f1", on_step=False, on_epoch=True, prog_bar=True)
            pl_module.log("train/precision", self.train_precision, metric_attribute="train_precision", on_step=False, on_epoch=True, prog_bar=True)
            pl_module.log("train/recall", self.train_recall, metric_attribute="train_recall", on_step=False, on_epoch=True, prog_bar=True)
            pl_module.log("train/cohen_kappa", self.train_kappa, metric_attribute="train_kappa", on_step=False, on_epoch=True, prog_bar=True)

            val_acc = self.val_acc.compute()
            pl_module.log("val/cls_loss", self.val_loss, metric_attribute="val_loss", on_step=False, on_epoch=True, prog_bar=True)
            pl_module.log("val/acc", val_acc, metric_attribute="val_acc", on_step=False, on_epoch=True, prog_bar=True)
            pl_module.log("val/f1", self.val_f1, metric_attribute="val_f1", on_step=False, on_epoch=True, prog_bar=True)
            pl_module.log("val/precision", self.val_precision, metric_attribute="val_precision", on_step=False, on_epoch=True, prog_bar=True)
            pl_module.log("val/recall", self.val_recall, metric_attribute="val_recall", on_step=False, on_epoch=True, prog_bar=True)
            pl_module.log("val/cohen_kappa", self.val_kappa, metric_attribute="val_kappa", on_step=False, on_epoch=True, prog_bar=True)

            self.val_acc_best = max(self.val_acc_best, val_acc)
            pl_module.log("val/acc_best", self.val_acc_best, on_step=False, on_epoch=True, prog_bar=True)

            if trainer.testing:
                self.test_acc.compute()
                self.test_f1.compute()
                self.test_precision.compute()
                self.test_recall.compute()
                self.test_kappa.compute()
                
                pl_module.log("test/cls_loss", self.test_loss, metric_attribute="test_loss", on_step=False, on_epoch=True, prog_bar=True)
                pl_module.log("test/acc", self.test_acc, metric_attribute="train_acc", on_step=False, on_epoch=True, prog_bar=True)
                pl_module.log("test/f1", self.test_f1, metric_attribute="test_f1", on_step=False, on_epoch=True, prog_bar=True)
                pl_module.log("test/precision", self.test_precision, metric_attribute="test_precision", on_step=False, on_epoch=True, prog_bar=True)
                pl_module.log("test/recall", self.test_recall, metric_attribute="test_recall", on_step=False, on_epoch=True, prog_bar=True)
                pl_module.log("test/cohen_kappa", self.test_kappa, metric_attribute="test_kappa", on_step=False, on_epoch=True, prog_bar=True)
            
            self.reset()
            