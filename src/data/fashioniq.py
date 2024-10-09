import json
from pathlib import Path

import torch
from lightning import LightningDataModule
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.transforms import transform_test, transform_train
from src.data.utils import pre_caption

Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombWarning


class FashionIQDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int = 4,
        pin_memory: bool = True,
        annotation: dict = {"train": "", "val": ""},
        targets: dict = {"train": "", "val": ""},
        img_dirs: dict = {"train": "", "val": ""},
        emb_dirs: dict = {"train": "", "val": ""},
        image_size: int = 384,
        **kwargs,  # type: ignore
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.transform_train = transform_train(image_size)
        self.transform_test = transform_test(image_size)

        self.data_train = FashionIQDataset(
            transform=self.transform_train,
            annotation=annotation["train"],
            targets=targets["train"],
            img_dir=img_dirs["train"],
            emb_dir=emb_dirs["train"],
            split="train",
        )
        self.data_val = FashionIQDataset(
            transform=self.transform_test,
            annotation=annotation["val"],
            targets=targets["val"],
            img_dir=img_dirs["val"],
            emb_dir=emb_dirs["val"],
            split="val",
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True,
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


class FashionIQTestDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        annotation: str,
        targets: str,
        img_dirs: str,
        emb_dirs: str,
        num_workers: int = 4,
        pin_memory: bool = True,
        image_size: int = 384,
        **kwargs,  # type: ignore
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.transform_test = transform_test(image_size)

        self.data_test = FashionIQDataset(
            transform=self.transform_test,
            annotation=annotation,
            targets=targets,
            img_dir=img_dirs,
            emb_dir=emb_dirs,
            split="test",
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
        )


class FashionIQDataset(Dataset):
    def __init__(
        self,
        transform,
        annotation: str,
        targets: str,
        img_dir: str,
        emb_dir: str,
        split: str,
        max_words: int = 30,
    ) -> None:
        super().__init__()

        self.transform = transform
        self.annotation_pth = annotation
        assert Path(annotation).exists(), f"Annotation file {annotation} does not exist"
        self.annotation = json.load(open(annotation, "r"))
        assert Path(targets).exists(), f"Targets file {targets} does not exist"
        self.targets = json.load(open(targets, "r"))
        self.target_ids = list(set(self.targets))
        self.target_ids.sort()

        self.split = split
        self.max_words = max_words
        self.img_dir = Path(img_dir)
        self.emb_dir = Path(emb_dir)
        assert split in [
            "train",
            "val",
            "test",
        ], f"Invalid split: {split}, must be one of train, val, or test"
        assert self.img_dir.exists(), f"Image directory {img_dir} does not exist"
        assert self.emb_dir.exists(), f"Embedding directory {emb_dir} does not exist"

        self.id2int = {id: i for i, id in enumerate(self.target_ids)}
        self.int2id = {i: id for i, id in enumerate(self.target_ids)}

        self.pairid2ref = {
            id: self.id2int[ann["candidate"]] for id, ann in enumerate(self.annotation)
        }
        self.pairid2tar = {
            id: self.id2int[ann["target"]] for id, ann in enumerate(self.annotation)
        }

        img_pths = self.img_dir.glob("*.png")
        emb_pths = self.emb_dir.glob("*.pth")
        self.id2imgpth = {img_pth.stem: img_pth for img_pth in img_pths}
        self.id2embpth = {emb_pth.stem: emb_pth for emb_pth in emb_pths}

        for ann in self.annotation:
            assert (
                ann["candidate"] in self.id2imgpth
            ), f"Path to image candidate {ann['candidate']} not found in {self.img_dir}"
            assert (
                ann["candidate"] in self.id2embpth
            ), f"Path to embedding candidate {ann['candidate']} not found in {self.emb_dir}"
            assert (
                ann["target"] in self.id2imgpth
            ), f"Path to image target {ann['target']} not found"
            assert (
                ann["target"] in self.id2embpth
            ), f"Path to embedding target {ann['target']} not found"

    def __len__(self) -> int:
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]

        reference_img_pth = self.id2imgpth[ann["candidate"]]
        reference_img = Image.open(reference_img_pth).convert("RGB")
        reference_img = self.transform(reference_img)

        cap1, cap2 = ann["captions"]
        caption = f"{cap1} and {cap2}"
        caption = pre_caption(caption, self.max_words)

        target_emb_pth = self.id2embpth[ann["target"]]
        target_feat = torch.load(target_emb_pth, weights_only=True).cpu()

        return {
            "ref_img": reference_img,
            "tar_img_feat": target_feat,
            "edit": caption,
            "pair_id": index,
        }
