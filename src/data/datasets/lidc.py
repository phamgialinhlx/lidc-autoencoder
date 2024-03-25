import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import imageio
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
        img = np.load(image_file)
        mask = np.load(mask_file)
        label = 1 if mask.sum() > 0 else 0

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
    i = 0
    # for i in range(12):
    print(ds[i]['data'].shape)
    print(ds[i]['mask'].shape)
    print("Mask Max", ds[i]['mask'].max())
    print("Mask Min", ds[i]['mask'].min())
    print("Mask Sum", ds[i]['mask'].sum())
    print(ds[i]['label'])
    # print(ds[0]['da
    image = ds[i]['data']
    image = image.permute(1, 2, 3, 0)
    image = (image + 1.0) * 127.5  # std + mean
    torch.clamp(image, 0, 255)
    image = image.cpu().numpy().astype(np.uint8)
    mask = ds[i]['mask']
    mask[mask == 1] = 255
    mask = mask.permute(1, 2, 3, 0)
    mask = mask.cpu().numpy().astype(np.uint8)
    frames = []
    t, h, w, c = image.shape
    # frame = torch.concat((x[0], y[0], pred[0]), dim=1)  # Concatenate images horizontally
    # frame = np.concatenate((x[0], y[0], pred[0]), axis=1)  # Concatenate images horizontally
    from IPython import embed; embed()
    for i in range(t):
        # Assuming x, y, and pred are images represented as numpy arrays
        frame = np.concatenate((image[i], mask[i]), axis=1)  # Concatenate images horizontally
        frames.append(frame)

    imageio.mimsave("./output.mp4", frames, fps=6)