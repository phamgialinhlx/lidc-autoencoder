from typing import Any, Dict, List, Optional, Tuple

import torch
import hydra
import numpy as np
import rootutils
from omegaconf import DictConfig

from lightning import Callback, LightningDataModule, LightningModule, Trainer

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.vq_gan_3d_module import VQGAN
from src.models.multihead_autoencoder_module import MultiheadVQGAN
from src.models.multihead_autoencoder_module import load_autoencoder
from torchmetrics import Dice, JaccardIndex, MaxMetric, MeanMetric, Accuracy, F1Score, Precision, Recall, CohenKappa
from IPython import embed
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

def get_a_scan(path: str):
    img = np.load(path)

    # range normalization to [-1, 1]
    img = (img - img.min()) / (img.max() - img.min())
    img = img * 2 - 1

    imageout = torch.from_numpy(img.copy()).float()
    imageout = imageout.unsqueeze(0)
    return imageout

@hydra.main(version_base="1.3", config_path="../configs", config_name="inference.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    cfg.ckpt_path = "./logs/train_autoencoder/runs/2024-04-05_13-19-13/lung-thesis/1coxuuhh/checkpoints/last.ckpt"
    cfg.ckpt_path = "./logs/train_autoencoder/runs/2024-04-06_00-46-28/lung-thesis/1s9bcfd0/checkpoints/last.ckpt"
    cfg.ckpt_path = "/work/hpc/pgl/lung-diffusion/logs/train_autoencoder/runs/2024-04-06_12-09-43/lung-thesis/35iyl5xo/checkpoints/epoch=138-step=446016.ckpt"
    # cfg.ckpt_path = "./logs/train_autoencoder/runs/2024-04-05_13-46-28/lung-thesis/3rc76tzt/checkpoints/last.ckpt"
    # cfg.ckpt_path = "./outputs/multihead_autoencoder_seg/lung-thesis/19wx27ka/checkpoints/epoch=64-step=157560.ckpt"
    # model = load_autoencoder(cfg.ckpt_path, "cuda")
    model = MultiheadVQGAN.load_from_checkpoint(cfg.ckpt_path, map_location="cuda")
    model.eval()

    use_ema = False
    task = "binary"
    num_classes = 1
    acc = Accuracy(task=task, num_classes=num_classes, average="micro")
    f1 = F1Score(task=task, num_classes=num_classes, average="micro")
    precision = Precision(task=task, num_classes=num_classes, average="micro")
    recall = Recall(task=task, num_classes=num_classes, average="micro")
    kappa = CohenKappa(task=task, num_classes=num_classes)
    
    # Initialize confusion matrix
    cm = np.zeros((2, 2))

    # Initialize TP, FP, FN, TN
    TP = FP = FN = TN = 0

    for i, batch in enumerate(tqdm(datamodule.val_dataloader())):
        # with torch.no_grad():
        batch['data'] = batch['data'].to('cuda')
        batch['mask'] = batch['mask'].to('cuda')
        batch['label'] = batch['label'].to('cuda')
        if use_ema:
            with model.ema_scope():
                _, preds, targets = model.forward_clasification(batch)
        else:
            _, preds, targets = model.forward_clasification(batch)

        # Compute confusion matrix
        pred_labels = preds.view(-1).cpu().numpy()
        true_labels = targets.view(-1).cpu().numpy()
        acc(preds.cpu(), targets.cpu())
        f1(preds.cpu(), targets.cpu())
        precision(preds.cpu(), targets.cpu())
        recall(preds.cpu(), targets.cpu())
        kappa(preds.cpu(), targets.cpu())
        
        cm += confusion_matrix(true_labels, pred_labels, labels=[0, 1])

    # Calculate TP, FP, FN, TN from confusion matrix
    TN, FP, FN, TP = cm.ravel()

    print("TP:", TP)
    print("FP:", FP)
    print("FN:", FN)
    print("TN:", TN)
    print("Accuracy", acc.compute())
    print("F1", f1.compute())
    print("Precision", precision.compute())
    print("Recall", recall.compute())
    print("CohenKappa", kappa.compute())
    embed()
    print("Segmentation part")
    jaccard = JaccardIndex(task="binary", num_classes=2)
    dice = Dice(ignore_index=0)
    # Initialize confusion matrix
    cm = np.zeros((2, 2))

    # Initialize TP, FP, FN, TN
    TP = FP = FN = TN = 0

    # Iterate over the train dataloader with tqdm
    for i, batch in enumerate(tqdm(datamodule.val_dataloader())):
        # with torch.no_grad():
        batch['data'] = batch['data'].to('cuda')
        batch['mask'] = batch['mask'].to('cuda')
        batch['label'] = batch['label'].to('cuda')
        if use_ema:
            with model.ema_scope():
                _, preds, targets = model.forward_segmentation(batch)
        else:
            _, preds, targets = model.forward_segmentation(batch)
        # preds = torch.tensor(np.load(f"multihead/predictions_{i}.npy")).to("cuda")
        B, C, T, W, H = preds.shape
        jaccard(preds.view(B, -1).to("cpu"), targets.to("cpu").view(B, -1))
        dice(preds.view(B, -1).to("cpu"), targets.to("cpu").view(B, -1))
        # Compute confusion matrix
        pred_labels = preds.view(-1).cpu().numpy()
        true_labels = batch['mask'].view(-1).cpu().numpy()
        cm += confusion_matrix(true_labels, pred_labels, labels=[0, 1])

        # Save each set of predictions with a unique filename
        # filename = f"multihead/predictions_{i}.npy"
        # np.save(filename, preds.cpu().numpy())
        # embed()

    # Calculate TP, FP, FN, TN from confusion matrix
    TN, FP, FN, TP = cm.ravel()

    print("TP:", TP)
    print("FP:", FP)
    print("FN:", FN)
    print("TN:", TN)
    print(jaccard.compute())
    print(dice.compute())
    embed()

if __name__ == "__main__":
    main()
