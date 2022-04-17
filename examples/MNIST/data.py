from multiprocessing import cpu_count
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, seed: int, batch_size: int):
        super().__init__()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.data_root = Path.cwd() / "data"
        self.seed = seed
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        MNIST(self.data_root, train=True, download=True)
        MNIST(self.data_root, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        transformations = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        if stage == "fit" or stage is None:
            mnist_train = MNIST(self.data_root, train=True, transform=transformations)
            self.train_dataset, self.val_dataset = random_split(
                mnist_train,
                [int(len(mnist_train) * 0.9), int(len(mnist_train) * 0.1)],
                generator=torch.Generator().manual_seed(self.seed),
            )
        if stage in ["test", "predict"] or stage is None:
            self.test_dataset = MNIST(
                self.data_root, train=False, transform=transformations
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=cpu_count()
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=cpu_count()
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=cpu_count()
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=cpu_count()
        )
