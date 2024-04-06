from typing import Any, Dict, List, Optional, Tuple

import torch
import hydra
import numpy as np
import rootutils
from omegaconf import DictConfig

from lightning import Callback, LightningDataModule, LightningModule, Trainer

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.vq_gan_3d_module import VQGAN
from src.models.diffusion_module import load_autoencoder

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
    # datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    # datamodule.setup()
    sample = get_a_scan("../LIDC-IDRI-Preprocessing/data/Image/LIDC-IDRI-0078/0078_NI001.npy")
    print('Shape of a sample:', sample.shape)
    sample = sample.unsqueeze(0)
    # model: LightningModule = hydra.utils.instantiate(cfg.model).to('cuda')
    model = load_autoencoder("./outputs/vq_gan_3d_low_compression/lung-thesis/2aglgm52/checkpoints/epoch=111-step=179200.ckpt", "cpu")
    # from IPython import embed; embed()
    encoder = model.encoder
    # z = model.pre_vq_conv(model.encoder(sample))
    # vq_output = model.codebook(z)
    # print(vq_output.keys())
    # print(vq_output["embeddings"].shape)
    # print(vq_output["encodings"].shape)
    print(encoder)
    print(sample.shape)
    h = encoder.conv_first(sample)
    print('conv_first', h.shape)
    for block in encoder.conv_blocks:
        h = block.down(h)
        h = block.res(h)
        print(h.shape)
    h = encoder.final_block(h)
    print(model.pre_vq_conv)
    print(h.shape)
    z = model.pre_vq_conv(h)
    print(z.shape)
    print(model.codebook)
    vq_output = model.codebook(z)
    print(vq_output['embeddings'].shape)
    print(model.post_vq_conv)
    print(model.decoder)
    h = model.post_vq_conv(vq_output['embeddings'])
    print(h.shape)
    h = model.decoder.final_block(h)
    print(h.shape)
    for i, block in enumerate(model.decoder.conv_blocks):
        h = block.up(h)
        h = block.res1(h)
        h = block.res2(h)
        print("block", i, h.shape)
    h = model.decoder.conv_last(h)
    print(model.decoder.conv_last)
    print(h.shape)
    # x_recon = model.decoder(model.post_vq_conv(vq_output['embeddings']))
    # sample = sample.cuda()
    # model = model.cuda()
    # recon_loss, x_recon, vq_output, perceptual_loss = model(sample)
    # print('vq_output["embeddings"].shape', vq_output["embeddings"].shape)
    # print('vq_output["encodings"].shape', vq_output["encodings"].shape)
    # print("x_recon.shape", x_recon.shape)

if __name__ == "__main__":
    main()
