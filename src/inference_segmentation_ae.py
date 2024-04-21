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
from src.models.vq_gan_3d_seg_head import VQGANSegHead
from src.models.utils.autoencoder import load_autoencoder_seg_head
from src.models.multihead_autoencoder_module import load_autoencoder
from torchmetrics import Dice, JaccardIndex, MaxMetric, MeanMetric, Accuracy, F1Score, Precision, Recall, CohenKappa
from IPython import embed
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import imageio

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
    cfg.ckpt_path = "./outputs/vq_gan_3d_seg_head/lung-thesis/2dyry4hr/checkpoints/epoch=115-step=187456.ckpt"
    # cfg.ckpt_path = "./logs/train_autoencoder/runs/2024-04-05_13-46-28/lung-thesis/3rc76tzt/checkpoints/last.ckpt"
    # cfg.ckpt_path = "./outputs/multihead_autoencoder_seg/lung-thesis/19wx27ka/checkpoints/epoch=64-step=157560.ckpt"
    # model = load_autoencoder(cfg.ckpt_path, "cuda")
    model = load_autoencoder_seg_head(cfg.ckpt_path, "cuda")
    model.eval()
    for i, batch in enumerate(tqdm(datamodule.val_dataloader())):
        x = batch["data"].cuda()
        seg = batch['segmentation']
        pred = model(x).squeeze(0)
        break

    x = (x + 1.0) * 127.5  # std + mean
    # x = x / 255.0
    x = x.squeeze(0)
    x = x.permute(1, 2, 3, 0)
    x = x.cpu().numpy()
    
    seg = (seg + 1.0) * 127.5  # std + mean
    torch.clamp(seg, 0, 255)
    # seg = seg / 255.0
    seg = seg.squeeze(0)
    seg = seg.permute(1, 2, 3, 0)
    seg = seg.cpu().numpy()
    
    embed()
    pred[pred < -1.0] = -1.0
    pred[pred > 1.0] = 1.0
    pred = (pred + 1.0) * 127.5  # std + mean
    # pred = pred / 255.0
    pred = pred.permute(1, 2, 3, 0)
    pred = pred.cpu().detach().numpy()
    frames = []
    t, h, w, c = x.shape
    # frame = torch.concat((x[0], y[0], pred[0]), dim=1)  # Concatenate images horizontally
    # frame = np.concatenate((x[0], y[0], pred[0]), axis=1)  # Concatenate images horizontally
    border_width = 5  # Width of the border in pixels

    for i in range(t):
        # Add a black border around each image
        x_padded = np.pad(x[i], ((border_width, border_width), (border_width, border_width), (0, 0)), mode='constant', constant_values=255)
        pred_padded = np.pad(pred[i], ((border_width, border_width), (border_width, border_width), (0, 0)), mode='constant', constant_values=255)
        if seg is not None:
            seg_padded = np.pad(seg[i], ((border_width, border_width), (border_width, border_width), (0, 0)), mode='constant', constant_values=255)
            # Now concatenate the images with borders
            frame = np.concatenate((x_padded, pred_padded, seg_padded), axis=1)
        else:
            # Now concatenate the images with borders
            frame = np.concatenate((x_padded, pred_padded), axis=1)  # Concatenate images horizontally with borders
        frames.append(frame.astype(np.uint8))

    imageio.mimsave("a.mp4", frames, fps=6)
if __name__ == "__main__":
    main()
