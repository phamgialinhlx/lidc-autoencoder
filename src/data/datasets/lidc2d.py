import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import imageio
import glob
import os

class LIDC2DDataset(Dataset):
    def __init__(
        self,
        root_dir='',
        transforms=None,
        nodules_only=False,
        include_origin_image=True,
        include_mask=False,
        include_segmentation=False
    ):
        super(LIDC2DDataset, self).__init__()
        self.paths = self.get_paths(os.path.join(root_dir, 'nodule'))
        if not nodules_only:
            self.paths += self.get_paths(os.path.join(root_dir, 'clean'))
        self.include_origin_image = include_origin_image
        self.include_mask = include_mask
        self.include_segmentation = include_segmentation
        self.transforms = transforms

    def __len__(self):
        return len(self.paths)

    def get_paths(self, dir):
        paths = []
        seg_path = os.path.join(dir, 'segmentation')
        seg_files = glob.glob(os.path.join(seg_path, '*.npy'), recursive=True)
        for seg in seg_files:
            paths.append((seg.replace('segmentation', 'img'), seg.replace('segmentation', 'mask'), seg))
        return paths

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
            img_seg = torch.from_numpy(img_seg.copy()).float()
            img_seg = img_seg.unsqueeze(0)

        imageout = torch.from_numpy(img.copy()).float()
        imageout = imageout.unsqueeze(0)
        if self.transforms is not None:
            # Apply additional transformations if provided
            imageout = self.transforms(imageout)
        out = {}
        if self.include_origin_image:
            out['data'] = imageout
        if self.include_mask:
            out['mask'] = mask
            out['label'] = label
        if self.include_segmentation:
            out['segmentation'] = img_seg
        return out

if __name__ == '__main__':
    ROOT_DIR = "/data/hpc/pgl/LIDC-IDRI-2D/data/"
    ds = LIDC2DDataset(
        root_dir=ROOT_DIR, 
        nodules_only=True,
        include_mask=True, 
        include_segmentation=True
    )
    
    i = 2829
    i = 9084
    i = 2011
    i = 0
    print("LIDC2DDataset class works well.")
    print("Length of the dataset is", ds.__len__())
    print('data', ds[i]['data'].shape)
    print('mask', ds[i]['mask'].shape)
    print('img_seg', ds[i]['segmentation'].shape)
    print("Mask Max", ds[i]['mask'].max())
    print("Mask Min", ds[i]['mask'].min())
    print("Mask Sum", ds[i]['mask'].sum())
    print(ds[i]['label'])
    image = ds[i]['data']
    # image = image.permute(1, 2, 3, 0)
    image = (image + 1.0) * 127.5  # std + mean
    torch.clamp(image, 0, 255)
    image = image.cpu().numpy().astype(np.uint8)
    mask = ds[i]['mask']
    mask[mask == 1] = 255
    # mask = mask.permute(1, 2, 3, 0)
    mask = mask.cpu().numpy().astype(np.uint8)
    img_seg = ds[i]['segmentation']
    # img_seg = img_seg.permute(1, 2, 3, 0)
    img_seg = (img_seg + 1.0) * 127.5  # std + mean
    torch.clamp(img_seg, 0, 255)
    img_seg = img_seg.cpu().numpy().astype(np.uint8)
    import cv2
    img = np.concatenate((image, mask, img_seg), axis=2)
    print(img[0].shape)
    cv2.imwrite("img.png", img[0])