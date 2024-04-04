from typing import Any, Dict, List, Optional, Tuple

import torch
import hydra
import numpy as np
import rootutils
from omegaconf import DictConfig

from lightning import Callback, LightningDataModule, LightningModule, Trainer

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.vq_gan_3d_module import VQGAN
from src.models.multihead_autoencoder_module import load_autoencoder
from torchmetrics import Dice, JaccardIndex, MaxMetric, MeanMetric
from tqdm import tqdm

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
    cfg.ckpt_path = "./logs/train autoencoder/runs/2024-04-04_19-02-23/lung-thesis/kychrjnk/checkpoints/last.ckpt"
    model = load_autoencoder(cfg.ckpt_path, "cuda")
    model.eval()

    jaccard = JaccardIndex(task="binary", num_classes=2)
    dice = Dice(num_classes=2, ignore_index=0, multiclass=True)

    # Iterate over the train dataloader with tqdm
    for batch in tqdm(datamodule.train_dataloader()):
        with torch.no_grad():
            batch['data'] = batch['data'].to('cuda')
            batch['mask'] = batch['mask'].to('cuda')
            batch['label'] = batch['label'].to('cuda')
            _, preds, _ = model.forward_segmentation(batch)
            B, C, T, W, H = preds.shape
            jaccard(preds.view(B, -1).to("cpu"), batch['mask'].int().to("cpu").view(B, -1))
            dice(preds.view(B, -1).to("cpu"), batch['mask'].int().to("cpu").view(B, -1))

    print(jaccard.compute())
    print(dice.compute())
    from IPython import embed
    embed()

if __name__ == "__main__":
    main()
