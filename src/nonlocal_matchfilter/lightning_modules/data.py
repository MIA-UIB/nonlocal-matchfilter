import logging

import lightning as L
from albumentations.core.composition import Compose
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset

logger = logging.getLogger(__name__)


class LitDataModule(L.LightningDataModule):
    def __init__(self, dataset: DictConfig, transforms: DictConfig, loader: DictConfig):
        super().__init__()
        self.datasets = dataset
        self.transforms = transforms
        self.loader = loader

    def setup(self, stage: str):
        match stage:
            case "fit":
                self._setup_train_val_dataset()
            case "test":
                self._setup_test_dataset()

    def _setup_train_val_dataset(self):
        # Assumes that the datasets are already split in training and validation data
        transforms_train = Compose(
            [self.transforms.train[tr] for tr in self.transforms.train.order],
            is_check_shapes=False,
        )
        transforms_val = Compose(
            [self.transforms.val[tr] for tr in self.transforms.val.order],
            is_check_shapes=False,
        )
        self.dataset_train = ConcatDataset(
            [
                self.datasets.train[dts](transforms=transforms_train)
                for dts in self.datasets.train.selected
            ]
        )
        self.dataset_val = ConcatDataset(
            [
                self.datasets.val[dts](transforms=transforms_val)
                for dts in self.datasets.val.selected
            ]
        )

    def _setup_test_dataset(self):
        transforms_test = Compose(
            [self.transforms.test[tr] for tr in self.transforms.test.order],
            is_check_shapes=False,
        )
        self.dataset_test = ConcatDataset(
            [
                self.datasets.test[dts](transforms=transforms_test)
                for dts in self.datasets.test.selected
            ]
        )

    def train_dataloader(self):
        return self.loader["train"](dataset=self.dataset_train)

    def val_dataloader(self):
        return self.loader["val"](dataset=self.dataset_val)

    def test_dataloader(self):
        return self.loader["test"](dataset=self.dataset_test)
