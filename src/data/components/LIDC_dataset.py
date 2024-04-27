import os 
import numpy as np 
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2
import glob

class LIDC_IDRI_Dataset(Dataset):
    def __init__(self, nodule_path, clean_path, mode, transforms=None, img_size=128):

        # nodule_path: path to dataset nodule image folder
        # clean_path: path to dataset clean image folder
        super().__init__()

        self.nodule_path = nodule_path
        self.clean_path = clean_path
        self.mode = mode
        self.transforms = transforms
        self.img_size = img_size

        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = Compose(
                [
                    A.Resize(self.img_size, self.img_size),
                    ToTensorV2(),
                ]
            )

        # define function to get list of (image, mask)
        self.file_list = self.get_paths(self.nodule_path, nodule=True)
        self.file_list += self.get_paths(self.clean_path, nodule=False)

        print(len(self.file_list))

    def __len__(self):
        return len(self.file_list)
    
    def get_paths(self, path, nodule=True):
        paths = []
        # files = glob.glob(os.path.join(path, '*.npy'), recursive=True)
        for file in path:
            if nodule:
                paths.append((file, file.replace("Image", "Mask").replace("NI", "MA")))
            else:
                paths.append((file, file.replace("Image", "Mask").replace("CN", "CM")))
        return paths 

    def __getitem__(self, index):
        image, mask = self.file_list[index]

        image = np.load(image)
        mask = np.load(mask).astype(np.int8)
        transformed = self.transforms(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"]
        image = image.to(torch.float)
        mask = mask.unsqueeze(0).to(torch.float)

        return {
            'segmentation': image,
            'mask': mask
        }

    def _normalize_image(self, image):
        min_val = np.min(image)
        max_val = np.max(image)

        if max_val - min_val > 0:
            image = (image - min_val) / (max_val - min_val)

        return image