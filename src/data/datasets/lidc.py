import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import imageio
import glob
import os

class LIDCDataset(Dataset):
    def __init__(self, root_dir='../LIDC', augmentation=False, transforms=None, mask_only=False, include_mask=False, include_segmentation=False):
        self.root_dir = root_dir
        self.paths = []
        self.paths = self.paths + self.get_paths(root_dir)
        if not mask_only:
            self.paths = self.paths + self.get_paths(os.path.join(root_dir, 'Clean')) 
        self.augmentation = augmentation
        self.transforms = transforms
        self.include_segmentation = include_segmentation
        self.paths = self.add_segmentation(root_dir)
        self.include_mask = include_mask

    def add_segmentation(self, dir):
        segmentation_path = os.path.join(dir, 'Image_Segmentation')
        paths = []
        for img, mask in self.paths:
            file = '/'.join(img.split('/')[-2:])
            paths.append((img, mask, os.path.join(segmentation_path, file)))
        return paths
    
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
        image_file, mask_file, segmentation_file = self.paths[index]
        img = np.load(image_file)
        # range normalization to [-1, 1]
        img = (img - img.min()) / (img.max() - img.min())
        img = img * 2 - 1
        
        if self.include_mask:
            mask = np.load(mask_file)
            mask = torch.from_numpy(mask.copy())
            mask = mask.unsqueeze(0)
            label = 1 if mask.sum() > 0 else 0

        if self.include_segmentation:
            img_seg = np.load(segmentation_file)
            # range normalization to [-1, 1]
            img_seg = (img_seg - img_seg.min()) / (img_seg.max() - img_seg.min())
            img_seg = img_seg * 2 - 1
            img_seg = torch.from_numpy(img_seg.copy()).float()
            img_seg = img_seg.unsqueeze(0)

        if self.augmentation:
            random_n = torch.rand(1)
            if random_n[0] > 0.5:
                img = np.flip(img, 2)
                if self.include_mask:
                    mask = np.flip(mask, 2)
                if self.include_segmentation:
                    img_seg = np.flip(img_seg, 2)

        imageout = torch.from_numpy(img.copy()).float()
        imageout = imageout.unsqueeze(0)
        if self.transforms is not None:
            # Apply additional transformations if provided
            imageout = self.transforms(imageout)
        out = {'data': imageout}
        if self.include_mask:
            out['mask'] = mask
            out['label'] = label
        if self.include_segmentation:
            out['segmentation'] = img_seg
        return out

if __name__ == "__main__":
    ds = LIDCDataset(root_dir='/work/hpc/pgl/LIDC-IDRI-Preprocessing/data/', mask_only=False, include_mask=True, include_segmentation=True)
    print('Number of samples in dataset:', ds.__len__())
    for i in range(1010):
        print('Sample:', ds.paths[i][0], i)
    i = 126
    print('Sample:', ds.paths[i][0])
    print('Sample:', ds.paths[i][1])
    print('Sample:', ds.paths[i][2])
    for j in range(ds.__len__()):
        path0 = ds.paths[j][0].split('/')[-1]
        path1 = ds.paths[j][1].split('/')[-1]
        path2 = ds.paths[j][2].split('/')[-1]
        if path0[:4] != path1[:4] and path1[:4] != path2[:4]:
            print(path0[:4])
            print(path1[:4])
            print(path2[:4])
        # print('Sample:', ds.paths[j][2], ds[j]['segmentation'].shape, ds[j]['segmentation'].sum())
        # if ds[j]['segmentation'].shape[-1] == 512:
            # print(ds.paths[j][2])
    # 607
    # for i in range(12):
    print('data', ds[i]['data'].shape)
    print('mask', ds[i]['mask'].shape)
    print('img_seg', ds[i]['segmentation'].shape)
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
    img_seg = ds[i]['segmentation']
    img_seg = img_seg.permute(1, 2, 3, 0)
    img_seg = (img_seg + 1.0) * 127.5  # std + mean
    torch.clamp(img_seg, 0, 255)
    img_seg = img_seg.cpu().numpy().astype(np.uint8)
    frames = []
    t, h, w, c = image.shape
    # frame = torch.concat((x[0], y[0], pred[0]), dim=1)  # Concatenate images horizontally
    # frame = np.concatenate((x[0], y[0], pred[0]), axis=1)  # Concatenate images horizontally

    # from IPython import embed; embed()
    for i in range(t):
        # Assuming x, y, and pred are images represented as numpy arrays
        frame = np.concatenate((image[i], mask[i], img_seg[i]), axis=1)  # Concatenate images horizontally
        frames.append(frame)

    imageio.mimsave("./output.mp4", frames, fps=6)
