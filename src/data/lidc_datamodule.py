from typing import Any, Dict, Optional, Tuple

import hydra
import torch
from lightning import LightningDataModule
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset, random_split
from omegaconf import DictConfig

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.datasets import LIDCDataset

class LIDCDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = 'data/lidc',
        batch_size: int = 1,
        image_size: int = 128,
        train_val_test_split: Tuple[int, int, int] = (10, 1, 1),
        augmentation: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # # data transformations
        t = []
        if image_size != 128:
            t.append(
                transforms.Resize(size = [image_size, image_size], antialias=True)
            )

        self.transforms = transforms.Compose(t)
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size


    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = LIDCDataset(root_dir=self.hparams.data_dir, transforms=self.transforms, augmentation=self.hparams.augmentation)
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass

@hydra.main(
    version_base="1.3", config_path="../../configs/data", config_name="lidc.yaml"
)
def main(cfg: DictConfig):
    print(cfg)
    cfg.image_size = 128
    cfg.train_val_test_split = (10, 1, 1)
    cfg.data_dir = "/mnt/work/Code/LIDC-IDRI-Preprocessing/data/"
    datamodule: LIDCDataModule = hydra.utils.instantiate(cfg)
    datamodule.setup()
    print('Number of samples in train set:', len(datamodule.train_dataloader()))
    print('Number of samples in validation set:', len(datamodule.val_dataloader()))
    print('Number of samples in test set:', len(datamodule.test_dataloader()))
    print('Shape of a sample:', datamodule.train_dataloader().dataset[0]['data'].shape)
    print('Data range: ', datamodule.train_dataloader().dataset[0]['data'].min(), datamodule.train_dataloader().dataset[0]['data'].max())
    print('Label:', datamodule.train_dataloader().dataset[0]['label'])
    print('Mask shape:', datamodule.train_dataloader().dataset[0]['mask'].shape)
    print('Mask range: ', datamodule.train_dataloader().dataset[0]['mask'].min(), datamodule.train_dataloader().dataset[0]['mask'].max())
if __name__ == "__main__":
    main()
