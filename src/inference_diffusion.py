from typing import Any, Dict, List, Optional, Tuple

import torch
import hydra
import numpy as np
import rootutils
from omegaconf import DictConfig

from lightning import Callback, LightningDataModule, LightningModule, Trainer

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# from src.models.vq_gan_3d_module import VQGAN
# from src.models.multihead_autoencoder_module import MultiheadVQGAN
from src.models.diffusion_module import DiffusionModule
from src.models.multihead_autoencoder_module import load_autoencoder
from torchmetrics import Dice, JaccardIndex, MaxMetric, MeanMetric, Accuracy, F1Score, Precision, Recall, CohenKappa
# from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import FrechetInceptionDistance
from sklearn.metrics import confusion_matrix
from torchvision.utils import make_grid, save_image
from IPython import embed
from tqdm import tqdm
import math

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
    cfg.data.batch_size = 8
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.data_train = None
    datamodule.data_val = None
    cfg.ckpt_path = "/work/hpc/pgl/lung-diffusion/outputs/ddpm_2d/lung-thesis/ybtohnfs/checkpoints/epoch=168-step=62083.ckpt"

    model = DiffusionModule.load_from_checkpoint(cfg.ckpt_path, map_location="cuda")
    model.eval()
    # use_ema = model.use_ema
    fid = FrechetInceptionDistance(feature=2048, normalize=True)
    device = "cuda"
    # model = model.to(device)
    data_key = "segmentation"

    # embed()
    for i, batch in enumerate(tqdm(datamodule.test_dataloader())):
        reals = batch[data_key].float().to(device)
        # fakes = batch[data_key].float().to(device)
        # embed()
        fakes = model.log_image(reals)
        # if use_ema:
            # with model.ema_scope():
        # else:
            # fakes = model.log_image(reals)

        reals = torch.cat([reals, reals, reals], dim=1)
        fakes = torch.cat([fakes, fakes, fakes], dim=1)

        # reals = torch.nn.functional.interpolate(reals,
        #                                         size=(299, 299),
        #                                         mode='bilinear')
        # fakes = torch.nn.functional.interpolate(fakes,
        #                                         size=(299, 299),
        #                                         mode='bilinear')
        
        # # model.to("cpu")
        fid.to(device)
        fid.update(reals.to(device), real=True)
        fid.update(fakes.to(device), real=False)
        fid.to(device)
        # model.to("cuda")
        # Save images
        # visualize(reals, 'reals.png')
        # visualize(fakes, 'fake.png')
        # embed() 

        break
    
    print(fid.compute())

if __name__ == "__main__":
    main()
