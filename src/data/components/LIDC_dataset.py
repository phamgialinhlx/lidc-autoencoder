import os 
import numpy as np 
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2

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
        self.file_list = self._get_file_list()

        print(len(self.file_list))

    def __len__(self):
        return len(self.file_list)
    
    def _get_file_list(self):
        file_list = []
        for dicom_path in self.nodule_path:
            
            # Get mask path of nodule image
            mask_path = dicom_path.replace("Image", "Mask")
            mask_path = mask_path.replace("NI", "MA")

            # Check whether mask path exist
            if os.path.exists(mask_path):

                image = np.load(dicom_path)

                # image = self._normalize_image(image)
                mask = np.load(mask_path).astype(np.uint8)

                # # convert image, mask to tensor

                # image = torch.from_numpy(image).to(torch.float)
                # mask = torch.from_numpy(mask).to(torch.float)

                # # add batch dimension 

                # image = image.unsqueeze(0)
                # mask = mask.unsqueeze(0)

                file_list.append((image, mask))
        
        for dicom_path in self.clean_path:
            # Get mask path of nodule image

            mask_path = dicom_path.replace("Image", "Mask")
            mask_path = mask_path.replace("CN", "CM")

            # Check whether mask path exist

            if os.path.exists(mask_path):

                image = np.load(dicom_path)

                # image = self._normalize_image(image)
                mask = np.load(mask_path).astype(np.uint8)

                # # convert image, mask to tensor

                # image = torch.from_numpy(image).to(torch.float)
                # mask = torch.from_numpy(mask).to(torch.float)

                # # add batch dimension 

                # image = image.unsqueeze(0)
                # mask = mask.unsqueeze(0)

                file_list.append((image, mask))

        return file_list

    def __getitem__(self, index):
        image, mask = self.file_list[index]

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