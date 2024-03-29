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

    def on_train_epoch_start(self, trainer, pl_module):
        self.train_acc.reset()
        self.train_f1.reset()
        self.train_precision.reset()
        self.train_recall.reset()
        self.train_kappa.reset()
        self.train_loss.reset()

    def on_validation_epoch_start(self, trainer, pl_module):
        self.val_acc.reset()
        self.val_f1.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_kappa.reset()
        self.val_loss.reset()

    def on_test_epoch_start(self, trainer, pl_module):
        self.test_acc.reset()
        self.test_f1.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_kappa.reset()
        self.test_loss.reset()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs["loss"]
        preds = outputs["preds"]
        targets = batch["label"]

        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.train_f1(preds, targets)
        self.train_precision(preds, targets)
        self.train_recall(preds, targets)
        self.train_kappa(preds, targets)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        loss = outputs["loss"]
        preds = outputs["preds"]
        targets = batch["label"]

        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.val_f1(preds, targets)
        self.val_precision(preds, targets)
        self.val_recall(preds, targets)
        self.val_kappa(preds, targets)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        loss = outputs["loss"]
        preds = outputs["preds"]
        targets = batch["label"]

        self.test_loss += loss.detach()
        self.test_acc(preds, targets)
        self.test_f1(preds, targets)
        self.test_precision(preds, targets)
        self.test_recall(preds, targets)
        self.test_kappa(preds, targets)

    def on_epoch_end(self, trainer, pl_module):
        pl_module.log("train/loss", self.train_loss / len(trainer.train_dataloader))
        pl_module.log("train/acc", self.train_acc.compute())
        pl_module.log("train/f1", self.train_f1.compute())
        pl_module.log("train/precision", self.train_precision.compute())
        pl_module.log("train/recall", self.train_recall.compute())
        pl_module.log("train/cohen_kappa", self.train_kappa.compute())

        val_acc = self.val_acc.compute()
        pl_module.log("val/loss", self.val_loss / len(trainer.val_dataloaders[0]))
        pl_module.log("val/acc", val_acc)
        pl_module.log("val/f1", self.val_f1.compute())
        pl_module.log("val/precision", self.val_precision.compute())
        pl_module.log("val/recall", self.val_recall.compute())
        pl_module.log("val/cohen_kappa", self.val_kappa.compute())

        self.val_acc_best = max(self.val_acc_best, val_acc)
        pl_module.log("val/acc_best", self.val_acc_best)

        if trainer.testing:
            pl_module.log("test/loss", self.test_loss / len(trainer.test_dataloaders[0]))
            pl_module.log("test/acc", self.test_acc.compute())
            pl_module.log("test/f1", self.test_f1.compute())
            pl_module.log("test/precision", self.test_precision.compute())
            pl_module.log("test/recall", self.test_recall.compute())
            pl_module.log("test/cohen_kappa", self.test_kappa.compute())