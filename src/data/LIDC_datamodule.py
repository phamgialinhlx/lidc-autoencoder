import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule
import torchvision
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from typing import Any, Dict, Optional, Tuple
import time
import csv
import glob

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.components.LIDC_dataset import LIDC_IDRI_Dataset


class LIDCDataModule(LightningDataModule):
    def __init__(
        self,
        nodule_dir: str = "/work/hpc/iai/loc/LIDC-IDRI-Preprocessing/data/Image",
        clean_dir: str = "/work/hpc/iai/loc/LIDC-IDRI-Preprocessing/data/Clean/Image",
        train_val_test_split: Tuple[int, int, int] = (3, 1, 1),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        num_nodule: int = 1000,
        num_clean: int = 1000,
        transforms: Any = None,
        img_size: int = 128,
        include_clean: bool = True,
        include_label: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # get all file_name in folder

        file_nodule_list = []
        file_clean_list = []

        # from IPython import embed
        # embed()
        # glob.glob(os.path.join(self.hparams.nodule_dir, '*/*.npy'), recursive=True)
        # get full path of each nodule file
        for root, _, files in os.walk(self.hparams.nodule_dir):
            for file in files:
                if file.endswith(".npy"):
                    dicom_path = os.path.join(root, file)
                    file_nodule_list.append(dicom_path)
        nodule_train, nodule_val, nodule_test = self.split_data(file_nodule_list, self.hparams.train_val_test_split)
        if include_clean:
            # get full path of each clean file
            for root, _, files in os.walk(self.hparams.clean_dir):
                for file in files:
                    if file.endswith(".npy"):
                        dicom_path = os.path.join(root, file)
                        file_clean_list.append(dicom_path)
            clean_train, clean_val, clean_test = self.split_data(file_clean_list, self.hparams.train_val_test_split)
        else:
            clean_train, clean_val, clean_test = [], [], []

        self.data_train = LIDC_IDRI_Dataset(nodule_train, clean_train, mode="train", transforms=self.hparams.transforms, img_size=img_size, include_label=include_label)

        self.data_val = LIDC_IDRI_Dataset(nodule_val, clean_val, mode="valid", transforms=self.hparams.transforms, img_size=img_size, include_label=include_label)

        self.data_test = LIDC_IDRI_Dataset(nodule_test, clean_test, mode="test", transforms=self.hparams.transforms, img_size=img_size, include_label=include_label)

    def split_data(self, file_paths, train_val_test_split):
        # get len files
        num_files = len(file_paths)

        # ratio
        train_ratio, val_ratio, test_ratio = train_val_test_split

        # get num train, val, test
        num_train = int(num_files * train_ratio / (train_ratio + val_ratio + test_ratio))
        num_val = int(num_files * val_ratio / (train_ratio + val_ratio + test_ratio))
        # get random index
        train_paths = list(np.random.choice(file_paths, num_train, replace=False))
        val_paths = list(np.random.choice(list(set(file_paths) - set(train_paths)), num_val, replace=False))
        test_paths = list(set(file_paths) - set(train_paths) - set(val_paths))
        return train_paths, val_paths, test_paths
    
    @property
    def num_classes(self):
        return 2

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass

if __name__ == "__main__":
    datamodule = LIDCDataModule(
        nodule_dir="/data/hpc/pgl/data/Image",
        clean_dir="/data/hpc/pgl/data/Clean/Image"
    )
    train_dataloader = datamodule.train_dataloader()
    batch_image = next(iter(train_dataloader))
    images, labels = batch_image
