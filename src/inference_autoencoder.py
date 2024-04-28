from typing import Any, Dict, List, Optional, Tuple

import torch
import hydra
import numpy as np
import rootutils
from omegaconf import DictConfig

from lightning import Callback, LightningDataModule, LightningModule, Trainer

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.swin_transformer_ae_module import SwinVQGAN
from src.models.vq_gan_2d_seg_module import VQGANSeg
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
    # datamodule.setup()
    cfg.ckpt_path = "/work/hpc/pgl/lung-diffusion/outputs/swin_transformer_ae_f48/checkpoints/epoch_036.ckpt"
    # cfg.ckpt_path = "./logs/train_autoencoder/runs/2024-04-05_13-46-28/lung-thesis/3rc76tzt/checkpoints/last.ckpt"
    # cfg.ckpt_path = "./outputs/multihead_autoencoder_seg/lung-thesis/19wx27ka/checkpoints/epoch=64-step=157560.ckpt"
    # model = load_autoencoder(cfg.ckpt_path, "cuda")
    model = SwinVQGAN.load_from_checkpoint(cfg.ckpt_path, map_location="cuda")
    input = None
    for i, batch in enumerate(tqdm(datamodule.test_dataloader())):
        input = batch["segmentation"].cuda()
        break

    visualize(input[:9], "rec_input.png")

    out, diff = model(input)
    visualize(out[:9], "swin_transformer_output.png")

    cfg.ckpt_path = "/work/hpc/pgl/lung-diffusion/outputs/vq_gan_2d_seg2/checkpoints/epoch_080.ckpt"
    model = VQGANSeg.load_from_checkpoint(cfg.ckpt_path, map_location="cuda")
    out, diff = model(input[:9])
    visualize(out[:9], "cnn_attention_output.png")
    embed()

if __name__ == "__main__":
    main()
