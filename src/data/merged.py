import numpy as np
import torch
from hydra.utils import instantiate
from lightning import LightningDataModule
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader


class MergedDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int = 4,
        pin_memory: bool = True,
        sampler_weights: str = "uniform",
        **kwargs,  # type: ignore
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        datasets_cfg = [kwargs[k] for k in kwargs if "dataset-" in k]
        assert len(datasets_cfg) > 0, "No datasets found"

        datasets_train = []
        datasets_val = []
        for dataset_cfg in datasets_cfg:
            assert "dataname" in dataset_cfg, "Dataset must have a dataname"
            print(f"Loading {dataset_cfg.dataname}")
            dataset_cfg = OmegaConf.create(dataset_cfg)
            dataset = instantiate(dataset_cfg)
            datasets_train.append(dataset.data_train)
            datasets_val.append(dataset.data_val)

        self.data_train = ConcatDataset(datasets_train)
        self.data_val = ConcatDataset(datasets_val)

        self.sampler = self.get_sampler(datasets_train, method=sampler_weights)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=True,
            sampler=self.sampler,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
        )

    @staticmethod
    def get_sampler(data_list, method="uniform"):
        total_length = sum(len(data) for data in data_list)

        if method == "uniform":
            weights = np.concatenate([np.ones(len(data)) for data in data_list])
        else:
            raise ValueError(f"Unknown method {method}")

        # Create a WeightedRandomSampler
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights, num_samples=total_length, replacement=False
        )

        return sampler
