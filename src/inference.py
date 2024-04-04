from typing import Any, Dict, List, Optional, Tuple

import hydra
import rootutils
from omegaconf import DictConfig
from lightning import Callback, LightningDataModule, LightningModule, Trainer

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    sample = datamodule.val_dataloader().dataset[0]['data']
    print('Shape of a sample:', sample.shape)
    model: LightningModule = hydra.utils.instantiate(cfg.model).to('cuda')

    sample = sample.unsqueeze(0).to('cuda')
    recon_loss, x_recon, vq_output, perceptual_loss = model(sample)
    print('vq_output["embeddings"].shape', vq_output["embeddings"].shape)
    print('vq_output["encodings"].shape', vq_output["encodings"].shape)
    print("x_recon.shape", x_recon.shape)

if __name__ == "__main__":
    main()
