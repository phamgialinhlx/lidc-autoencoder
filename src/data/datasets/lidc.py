import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import glob
import os

class LIDCDataset(Dataset):
    def __init__(self, root_dir='../LIDC', augmentation=False, transforms=None):
        self.root_dir = root_dir
        self.paths = []
        self.paths = self.paths + self.get_paths(root_dir) + self.get_paths(os.path.join(root_dir, 'Clean'))
        self.augmentation = augmentation
        self.transforms = transforms

    def get_paths(self, dir):
        image_path = os.path.join(dir, 'Image')
        mask_path = os.path.join(dir, 'Mask')
        image_files = glob.glob(os.path.join(image_path, '**/*.npy'), recursive=True)
        mask_files = glob.glob(os.path.join(mask_path, '**/*.npy'), recursive=True)
        paths = []
        for img, mask in zip(image_files, mask_files):
            paths.append((img, mask))
        return paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image_file, mask_file = self.paths[index]
        if image_file.__contains__('Clean'):
            label = 0
        else:
            label = 1
        img = np.load(image_file)
        mask = np.load(mask_file)

        # range normalization to [-1, 1]
        img = (img - img.min()) / (img.max() - img.min())
        img = img * 2 - 1

        if self.augmentation:
            random_n = torch.rand(1)
            if random_n[0] > 0.5:
                img = np.flip(img, 2)
        
        imageout = torch.from_numpy(img.copy()).float()
        imageout = imageout.unsqueeze(0)
        mask = torch.from_numpy(mask.copy())
        mask = mask.unsqueeze(0)
        
        if self.transforms is not None:
            # Apply additional transformations if provided
            imageout = self.transforms(imageout)

        return {'data': imageout, 'mask': mask, 'label': label}

if __name__ == "__main__":
    ds = LIDCDataset(root_dir='/mnt/work/Code/LIDC-IDRI-Preprocessing/data/')
    print('Number of samples in dataset:', ds.__len__() )
    print(ds[0]['data'].shape)
    print(ds[0]['mask'].shape)
    print(ds[0]['label'])
    