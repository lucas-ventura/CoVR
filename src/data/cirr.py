import json
from pathlib import Path

import torch
from lightning import LightningDataModule
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.transforms import transform_test, transform_train
from src.data.utils import pre_caption

Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombWarning


class CIRRDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int = 4,
        pin_memory: bool = True,
        annotation: dict = {"train": "", "val": ""},
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

        self.data_train = CIRRDataset(
            transform=self.transform_train,
            annotation=annotation["train"],
            img_dir=img_dirs["train"],
            emb_dir=emb_dirs["train"],
            split="train",
        )
        self.data_val = CIRRDataset(
            transform=self.transform_test,
            annotation=annotation["val"],
            img_dir=img_dirs["val"],
            emb_dir=emb_dirs["val"],
            split="val",
        )

    def prepare_data(self):
        # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
        # download data, pre-process, split, save to disk, etc...
        pass

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


class CIRRTestDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        annotation: str,
        img_dirs: str,
        emb_dirs: str,
        num_workers: int = 4,
        pin_memory: bool = True,
        image_size: int = 384,
        split: str = "test",
        **kwargs,  # type: ignore
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.transform_test = transform_test(image_size)

        self.data_test = CIRRDataset(
            transform=self.transform_test,
            annotation=annotation,
            img_dir=img_dirs,
            emb_dir=emb_dirs,
            split=split,
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


class CIRRDataset(Dataset):
    def __init__(
        self,
        transform,
        annotation: str,
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

        self.pairid2ref = {ann["pairid"]: ann["reference"] for ann in self.annotation}
        self.pairid2members = {
            ann["pairid"]: ann["img_set"]["members"] for ann in self.annotation
        }
        if "test" not in Path(self.annotation_pth).stem:
            self.pairid2tar = {
                ann["pairid"]: ann["target_hard"] for ann in self.annotation
            }
        else:
            self.pairid2tar = None

        if split == "train":
            img_pths = self.img_dir.glob("*/*.png")
            emb_pths = self.emb_dir.glob("*/*.pth")
        else:
            img_pths = self.img_dir.glob("*.png")
            emb_pths = self.emb_dir.glob("*.pth")
        self.id2imgpth = {img_pth.stem: img_pth for img_pth in img_pths}
        self.id2embpth = {emb_pth.stem: emb_pth for emb_pth in emb_pths}

        for ann in self.annotation:
            assert (
                ann["reference"] in self.id2imgpth
            ), f"Path to reference {ann['reference']} not found in {self.img_dir}"
            assert (
                ann["reference"] in self.id2embpth
            ), f"Path to reference {ann['reference']} not found in {self.emb_dir}"
            if split != "test":
                assert (
                    ann["target_hard"] in self.id2imgpth
                ), f"Path to target {ann['target_hard']} not found"
                assert (
                    ann["target_hard"] in self.id2embpth
                ), f"Path to target {ann['target_hard']} not found"

    def __len__(self) -> int:
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]

        reference_img_pth = self.id2imgpth[ann["reference"]]
        reference_img = Image.open(reference_img_pth).convert("RGB")
        reference_img = self.transform(reference_img)

        caption = pre_caption(ann["caption"], self.max_words)

        if self.split == "test":
            return {
                "ref_img": reference_img,
                "edit": caption,
                "pair_id": ann["pairid"],
            }

        target_emb_pth = self.id2embpth[ann["target_hard"]]
        target_feat = torch.load(target_emb_pth, weights_only=True).cpu()

        return_dict = {
            "ref_img": reference_img,
            "tar_img_feat": target_feat,
            "edit": caption,
            "pair_id": ann["pairid"],
        }

        return return_dict
