import random
from pathlib import Path

import pandas as pd
import torch
from lightning import LightningDataModule
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.transforms import transform_test, transform_train
from src.data.utils import pre_caption
from src.data.webvid_covr import WebVidCoVRDataset
from src.tools.files import write_txt
from src.tools.utils import print_dist

Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombWarning


class CCCoIRDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int = 4,
        pin_memory: bool = True,
        annotation: dict = {"train": "", "val": ""},
        img_dirs: dict = {"train": "", "val": ""},
        emb_dirs: dict = {"train": "", "val": ""},
        image_size: int = 384,
        si_tc_weight=0,
        **kwargs,  # type: ignore
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.transform_train = transform_train(image_size)
        self.transform_test = transform_test(image_size)

        self.data_train = CCCoIRDataset(
            transform=self.transform_train,
            annotation=annotation["train"],
            img_dir=img_dirs["train"],
            emb_dir=emb_dirs["train"],
            split="train",
            si_tc_weight=si_tc_weight,
            image_size=image_size,
        )

        self.data_val = WebVidCoVRDataset(
            transform=self.transform_test,
            annotation=annotation["val"],
            vid_dir=img_dirs["val"],
            emb_dir=emb_dirs["val"],
            split="val",
            emb_pool="query",
            iterate="pth2",
            vid_query_method="middle",
            vid_frames=1,
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


class CCCoIRTestDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        annotation: str,
        img_dir: str,
        emb_dir: str,
        num_workers: int = 4,
        pin_memory: bool = True,
        image_size: int = 384,
        iterate: str = "pth2",
        **kwargs,  # type: ignore
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.iterate = iterate

        self.transform_test = transform_test(image_size)

        self.data_test = CCCoIRDataset(
            transform=self.transform_test,
            annotation=annotation,
            img_dir=img_dir,
            emb_dir=emb_dir,
            split="test",
            iterate=self.iterate,
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


class CCCoIRDataset(Dataset):
    def __init__(
        self,
        transform,
        annotation: str,
        img_dir: str,
        emb_dir: str,
        split: str,
        max_words: int = 30,
        iterate: str = "pth2",
        si_tc_weight=0,
        image_size: int = 384,
    ) -> None:
        super().__init__()

        self.transform = transform
        self.annotation_pth = Path(annotation)
        self.image_size = image_size
        assert (
            self.annotation_pth.exists()
        ), f"Annotation file {annotation} does not exist"
        self.df = pd.read_csv(annotation)

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

        if split == "train":
            img_pths = list(self.img_dir.glob("*/*.png")) + list(
                self.img_dir.glob("*/*.jpg")
            )
            emb_pths = self.emb_dir.glob("*/*.pth")
            id2imgpth = {
                img_pth.parent.stem + "/" + img_pth.stem: img_pth
                for img_pth in img_pths
            }
            id2embpth = {
                emb_pth.parent.stem + "/" + emb_pth.stem: emb_pth
                for emb_pth in emb_pths
            }
        else:
            img_pths = list(self.img_dir.glob("*.png")) + list(
                self.img_dir.glob("*.jpg")
            )
            emb_pths = self.emb_dir.glob("*.pth")
            id2imgpth = {img_pth.stem: img_pth for img_pth in img_pths}
            id2embpth = {emb_pth.stem: emb_pth for emb_pth in emb_pths}

        assert len(id2imgpth) > 0, f"No videos found in {img_dir}"
        assert len(id2embpth) > 0, f"No embeddings found in {emb_dir}"

        print(f"Found {len(id2imgpth)} images in {img_dir}")
        print(f"Found {len(id2embpth)} embeddings in {emb_dir}")

        self.df["path1"] = self.df["pth1"].apply(lambda x: id2imgpth.get(x, None))  # type: ignore
        self.df["path2"] = self.df["pth2"].apply(lambda x: id2embpth.get(x, None))  # type: ignore

        # Count unique missing paths
        missing_pth1 = self.df[self.df["path1"].isna()]["pth1"].unique().tolist()
        missing_pth1.sort()
        total_pth1 = self.df["pth1"].nunique()
        assert len(missing_pth1) != total_pth1, "Missing all pth1's"

        missing_pth2 = self.df[self.df["path2"].isna()]["pth2"].unique().tolist()
        missing_pth2.sort()
        total_pth2 = self.df["pth2"].nunique()
        assert len(missing_pth2) != total_pth2, "Missing all pth2's"

        if len(missing_pth1) > 0:
            print_dist(
                f"Missing {len(missing_pth1)} pth1's ({len(missing_pth1)/total_pth1 * 100:.1f}%), saving them to missing_pth1-{split}.txt"
            )
            if split == "test":
                raise ValueError(
                    f"Missing {len(missing_pth1)} pth1's ({len(missing_pth1)/total_pth1 * 100:.1f}%) in test split"
                )
            write_txt(missing_pth1, f"missing_pth1-{split}.txt")
        if len(missing_pth2) > 0:
            print_dist(
                f"Missing {len(missing_pth2)} pth2's ({len(missing_pth2)/total_pth2 * 100:.1f}%), saving them to missing_pth2-{split}.txt"
            )
            if split == "test":
                raise ValueError(
                    f"Missing {len(missing_pth2)} pth2's ({len(missing_pth2)/total_pth2 * 100:.1f}%) in test split"
                )
            write_txt(missing_pth2, f"missing_pth2-{split}.txt")

        # Remove missing paths
        self.df = self.df[self.df["path1"].notna()]
        self.df = self.df[self.df["path2"].notna()]
        self.df.reset_index(drop=True, inplace=True)

        self.max_words = max_words

        if iterate in ["idx", "triplets"]:
            iterate = "idx"
            self.df["idx"] = self.df.index
        self.iterate = iterate
        self.target_txts = self.df[iterate].unique()
        assert (
            iterate in self.df.columns
        ), f"{iterate} not in {self.annotation_pth.stem}"
        self.df.sort_values(iterate, inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        pths = set(self.df["pth1"].unique()) | set(self.df["pth2"].unique())
        pths = sorted(list(pths))
        self.id2int = {pth: i for i, pth in enumerate(pths)}
        self.int2id = {i: pth for i, pth in enumerate(pths)}
        assert len(self.id2int) == len(self.int2id), "id2int and int2id are not equal"
        self.df["int1"] = self.df["pth1"].apply(lambda x: self.id2int[x])
        self.df["int2"] = self.df["pth2"].apply(lambda x: self.id2int[x])
        self.pairid2ref = self.df["int1"].to_dict()
        assert (
            self.df["int1"].nunique() == self.df["pth1"].nunique()
        ), "int1 is not unique"
        assert (
            self.df["int2"].nunique() == self.df["pth2"].nunique()
        ), "int2 is not unique"

        self.pairid2tar = self.df["int2"].to_dict()
        self.df.set_index(iterate, inplace=True)
        self.df[iterate] = self.df.index

        if split == "test":
            assert (
                len(self.target_txts) == self.df.shape[0]
            ), "Test split should have one caption per row"

        # Check if text embeddings exist
        self.txt2emb = None
        if si_tc_weight > 0:
            txt2emb_pth = Path(emb_dir) / f"../txt2_{self.annotation_pth.stem}.pth"
            if "blip2" in str(txt2emb_pth):
                model = "blip2"
            elif "blip" in str(txt2emb_pth):
                model = "blip"
            elif "clip" in str(txt2emb_pth):
                model = "clip"
            else:
                raise ValueError(f"Invalid model: {txt2emb_pth}")
            assert txt2emb_pth.exists(), f"txt2emb does not exist: {txt2emb_pth}. Please compute them with: python tools/embs/save_{model}_embs_txts.py {self.annotation_pth} {self.emb_dir.parent}"
            self.txt2emb = torch.load(txt2emb_pth, weights_only=True)
            assert len(self.txt2emb["texts"]) == len(
                self.txt2emb["feats"]
            ), "txt2emb is not valid"
            self.txt2emb = {
                txt: feat
                for txt, feat in zip(self.txt2emb["texts"], self.txt2emb["feats"])
            }
            txt2s = set(self.df["txt2"].unique().tolist())
            assert txt2s.issubset(
                set(self.txt2emb.keys())
            ), "txt2emb does not contain all txt2's"

    def __len__(self) -> int:
        return len(self.target_txts)

    def __getitem__(self, index):
        target_txt = self.target_txts[index]
        ann = self.df.loc[target_txt]
        if ann.ndim > 1:
            ann = ann.sample()
            ann = ann.iloc[0]

        reference_img_pth = str(ann["path1"])
        try:
            reference_img = Image.open(reference_img_pth).convert("RGB")
            reference_img = self.transform(reference_img)
        except Exception as e:
            print(f"Error opening {reference_img_pth}: {e}")
            reference_img = torch.zeros(3, self.image_size, self.image_size)

        edit = ann["edit"]
        if isinstance(edit, list):
            edit = random.choice(edit)
        caption = pre_caption(edit, self.max_words)

        target_pth = str(ann["path2"])
        target_feat = torch.load(target_pth, weights_only=True).cpu()

        return_dict = {
            "ref_img": reference_img,
            "tar_img_feat": target_feat,
            "edit": caption,
            "pair_id": index,
        }

        if self.txt2emb is not None:
            return_dict["tar_txt_feat"] = self.txt2emb[ann["txt2"]]

        return return_dict
