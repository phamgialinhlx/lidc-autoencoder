from typing import Any, Dict, List, Optional, Tuple

import torch
import hydra
import numpy as np
import rootutils
from omegaconf import DictConfig

from lightning import Callback, LightningDataModule, LightningModule, Trainer

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.multihead_swin_transformer_ae_module import MultiheadSwinVQGAN
from src.models.multihead_autoencoder_module import load_autoencoder
from torchmetrics import Dice, JaccardIndex, MaxMetric, MeanMetric, Accuracy, F1Score, Precision, Recall, CohenKappa
from IPython import embed
import math
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from torchvision.utils import make_grid, save_image

def get_a_scan(path: str):
    img = np.load(path)

    # range normalization to [-1, 1]
    img = (img - img.min()) / (img.max() - img.min())
    img = img * 2 - 1

    imageout = torch.from_numpy(img.copy()).float()
    imageout = imageout.unsqueeze(0)
    return imageout

def visualize(img, file_name):
    nrows = math.ceil(math.sqrt(img.shape[0]))

    value_range = (0, 1)

    img = make_grid(
        img, nrow=nrows, normalize=True, value_range=value_range
    )

    # Save images
    save_image(img, file_name)

@hydra.main(version_base="1.3", config_path="../configs", config_name="inference.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    cfg.ckpt_path = "/work/hpc/pgl/lung-diffusion/logs/train_autoencoder/runs/swin_cls_seg/lung-thesis/3h5zryug/checkpoints/epoch=29-step=62280.ckpt"
    # cfg.ckpt_path = "./logs/train_autoencoder/runs/2024-04-05_13-46-28/lung-thesis/3rc76tzt/checkpoints/last.ckpt"
    # cfg.ckpt_path = "./outputs/multihead_autoencoder_seg/lung-thesis/19wx27ka/checkpoints/epoch=64-step=157560.ckpt"
    # model = load_autoencoder(cfg.ckpt_path, "cuda")
    model = MultiheadSwinVQGAN.load_from_checkpoint(cfg.ckpt_path, map_location="cuda")
    model.eval()

    input = None

    for i, batch in enumerate(tqdm(datamodule.val_dataloader())):
        input = batch
        break
    
    input["segmentation"] = input["segmentation"].cuda()
    logits = model.segmentation_decoder(model.encoder, input["segmentation"], model.encoder_normalize)
    if model.segmentation_decoder.out_channels == 2:
        preds = torch.argmax(logits, dim=1).unsqueeze(0)
        preds = preds.permute(1, 0, 2, 3)
    else:
        preds = torch.sigmoid(logits)
        preds[preds >= 0.5] = 1
        preds[preds < 0.5] = 0

    visualize(input["segmentation"][:9], "mask_input.png")
    visualize(input["mask"][:9], "mask_gt.png")
    visualize(preds[:9].float(), "mask_beta_1.png")

    cfg.ckpt_path = "/work/hpc/pgl/lung-diffusion/logs/train_autoencoder/runs/2024-05-10_22-08-00/lung-thesis/oruicgly/checkpoints/epoch=36-step=76812.ckpt"

    model = MultiheadSwinVQGAN.load_from_checkpoint(cfg.ckpt_path, map_location="cuda")
    logits = model.segmentation_decoder(model.encoder, input["segmentation"], model.encoder_normalize)
    if model.segmentation_decoder.out_channels == 2:
        preds = torch.argmax(logits, dim=1).unsqueeze(0)
        preds = preds.permute(1, 0, 2, 3)
    else:
        preds = torch.sigmoid(logits)
        preds[preds >= 0.5] = 1
        preds[preds < 0.5] = 0

    visualize(preds[:9].float(), "mask_beta_25.png")

if __name__ == "__main__":
    main()
