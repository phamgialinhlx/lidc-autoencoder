import torch
from vit_pytorch import ViT
import torch.nn as nn
import torch.nn.functional as F

class VIT(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.first_convs = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 3, kernel_size=3, padding=1)
        )
        self.vit = ViT(
            image_size = 32,
            patch_size = 4,
            num_classes = 2,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )

    def forward(self, x):
        x = self.first_convs(x)
        return self.vit(x)
