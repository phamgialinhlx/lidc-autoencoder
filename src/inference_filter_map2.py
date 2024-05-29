from typing import Any, Dict, List, Optional, Tuple

import math
import torch
import hydra
import numpy as np
import rootutils
from omegaconf import DictConfig
from IPython import embed
from torchvision.utils import make_grid, save_image
from lightning import Callback, LightningDataModule, LightningModule, Trainer
import matplotlib.pyplot as plt
import os

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.ds_segmentation_module import DownstreamSegmentationModule

def visualize(img, file_name):
    nrows = math.ceil(math.sqrt(img.shape[0]))

    value_range = (0, 1)

    img = make_grid(
        img, nrow=nrows, normalize=True, value_range=value_range
    )

    # Save images
    save_image(img, file_name)


def save_feature_maps(feature_map, rows, cols, output_dir='feature_maps', file_name='feature_maps_grid.png'):
    """
    Save feature maps in a grid layout.

    Parameters:
    - feature_map: NumPy array of shape (1, C, H, W)
    - rows: Number of rows in the grid
    - cols: Number of columns in the grid
    - output_dir: Directory to save the image
    - file_name: Name of the saved image file
    """
    # Squeeze to remove the batch dimension
    feature_map = np.squeeze(feature_map, axis=0)
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    n_channels = feature_map.shape[0]
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    
    for i in range(rows * cols):
        row = i // cols
        col = i % cols
        ax = axes[row, col]
        
        if i < n_channels:
            ax.imshow(feature_map[i], cmap='viridis')
            ax.set_title(f'Channel {i+1}')
        else:
            ax.axis('off')  # Turn off axes for empty subplots
        
        ax.axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, file_name)
    plt.savefig(output_path)
    plt.close()

    print(f"Feature maps saved to: {output_path}")

def save_tensors_as_numpy(tensor_list, output_dir='numpy_arrays', base_file_name='tensor_array'):
    """
    Convert a list of PyTorch tensors to NumPy arrays and save them.

    Parameters:
    - tensor_list: List of PyTorch tensors
    - output_dir: Directory to save the NumPy arrays
    - base_file_name: Base name for the saved NumPy files
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    for i, tensor in enumerate(tensor_list):
        # Convert tensor to NumPy array
        if tensor.is_cuda:
            tensor = tensor.cpu()
        numpy_array = tensor.numpy()
        
        # Save the NumPy array
        file_name = f"{base_file_name}_{i+1}.npy"
        output_path = os.path.join(output_dir, file_name)
        np.save(output_path, numpy_array)
        
        print(f"Saved {output_path}")

@hydra.main(version_base="1.3", config_path="../configs", config_name="inference.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    # datamodule.setup()
    ckpt_path = "/work/hpc/pgl/lung-diffusion/outputs/downstream_unetr_soft_dice_v2/lung-thesis/1axptv6r/checkpoints/epoch=74-step=19500.ckpt"
    model: LightningModule = hydra.utils.instantiate(cfg.model).to('cuda')
    model = DownstreamSegmentationModule.load_from_checkpoint(ckpt_path, net=model.net, criterion=model.criterion)
    for i, batch in enumerate(datamodule.test_dataloader()):
        batch = batch
        break
    batch["segmentation"] = batch["segmentation"].cuda()
    batch["mask"] = batch["mask"].cuda()
    INDEX_SAMPLE = 1
    sample = batch["segmentation"][INDEX_SAMPLE].unsqueeze(0)

    encoder = model.net.swin_encoder.swinViT
    output = encoder(sample)
    
    FM_DIR = "downstream_unetr_soft_dice_v2"
    os.makedirs(FM_DIR, exist_ok=True)

    # 0 torch.Size([1, 48, 64, 64])                                                                                                                                                      │········
    # 1 torch.Size([1, 96, 32, 32])                                                                                                                                                      │········
    # 2 torch.Size([1, 192, 16, 16])                                                                                                                                                     │········
    # 3 torch.Size([1, 384, 8, 8])                                                                                                                                                       │········
    # 4 torch.Size([1, 768, 4, 4]) 
    save_tensors_as_numpy(output, output_dir=FM_DIR, base_file_name='feature_maps')
    save_feature_maps(output[0].cpu().detach().numpy(), rows=7, cols=7, output_dir=FM_DIR, file_name='feature_maps_grid_0.png')
    save_feature_maps(output[1].cpu().detach().numpy(), rows=8, cols=12, output_dir=FM_DIR, file_name='feature_maps_grid_1.png')
    save_feature_maps(output[2].cpu().detach().numpy(), rows=16, cols=12, output_dir=FM_DIR, file_name='feature_maps_grid_2.png')
    save_feature_maps(output[3].cpu().detach().numpy(), rows=16, cols=24, output_dir=FM_DIR, file_name='feature_maps_grid_3.png')
    save_feature_maps(output[4].cpu().detach().numpy(), rows=32, cols=24, output_dir=FM_DIR, file_name='feature_maps_grid_4.png')
    for i, o in enumerate(output):
        print(i, o.shape)
        break
        # o = torch.clamp(o, 0, 1)
        # mean_feature_map = torch.sum(o, 1) / o.shape[1]
        # mean_feature_map = torch.clamp(mean_feature_map, 0, 1)
        # mean_feature_map = mean_feature_map.unsqueeze(1)
        # print(i, mean_feature_map.shape)

    visualize(batch["mask"][INDEX_SAMPLE].unsqueeze(0), f"{FM_DIR}/mask.png")

    embed()

if __name__ == "__main__":
    main()
