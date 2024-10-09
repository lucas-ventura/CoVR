import json
from pathlib import Path
from typing import Dict, List, Literal, Union

import torch
from lightning import LightningDataModule
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.data.transforms import transform_test
from src.data.utils import pre_caption

Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombWarning


class CIRCOTestDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        split: Literal["val", "test"],
        data_path: str,
        emb_dir: str,
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

        self.data_test = CIRCODataset(
            transform=self.transform_test,
            data_path=data_path,
            emb_dir=emb_dir,
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


class CIRCODataset(Dataset):
    """
    CIRCO dataset, code adapted from miccunifi/CIRCO
    """

    def __init__(
        self,
        transform,
        data_path: Union[str, Path],
        emb_dir: str,
        split: Literal["val", "test"],
        max_words: int = 30,
    ) -> None:
        """
        Args:
            transform (callable): function which preprocesses the image
            data_path (Union[str, Path]): path to CIRCO dataset
            split (str): dataset split, should be in ['test', 'val']
        """

        self.transform = transform
        data_path = Path(data_path)
        assert data_path.exists(), f"Annotation file {data_path} does not exist"
        self.split = split
        self.max_words = max_words
        img_dir = data_path / "COCO2017_unlabeled" / "unlabeled2017"
        self.img_dir = Path(img_dir)
        self.emb_dir = Path(emb_dir)
        assert split in [
            "val",
            "test",
        ], f"Invalid split: {split}, must be one of val or test"
        assert self.img_dir.exists(), f"Image directory {img_dir} does not exist"
        assert self.emb_dir.exists(), f"Embedding directory {emb_dir} does not exist"

        # Load COCO images information
        with open(
            data_path
            / "COCO2017_unlabeled"
            / "annotations"
            / "image_info_unlabeled2017.json",
            "r",
        ) as f:
            imgs_info = json.load(f)

        self.img_paths = [
            img_dir / img_info["file_name"] for img_info in imgs_info["images"]
        ]
        self.img_ids = [img_info["id"] for img_info in imgs_info["images"]]
        self.img_ids_indexes_map = {
            str(img_id): i for i, img_id in enumerate(self.img_ids)
        }

        # get CIRCO annotations
        with open(data_path / "annotations" / f"{split}.json", "r") as f:
            self.annotations: List[dict] = json.load(f)

        # Get maximum number of ground truth images (for padding when loading the images)
        self.max_num_gts = 23  # Maximum number of ground truth images

        # Get the embeddings
        emb_pth = self.emb_dir / "all_embs.pt"
        if emb_pth.exists():
            embs_dict = torch.load(emb_pth, weights_only=True)
            self.embs = embs_dict["embs"]
            assert self.img_ids == embs_dict["ids"], "Image IDs do not match"
        else:
            emb_pths = list(self.emb_dir.glob("*.pth"))
            assert (
                len(emb_pths) == len(self.img_ids)
            ), f"Number of embeddings {len(emb_pths)} does not match number of images {len(self.img_ids)}"
            emb_pths = list(self.emb_dir.glob("*.pth"))
            img_id2emb_pth = {int(p.stem): p for p in emb_pths}
            embs = [
                torch.load(img_id2emb_pth[img_id], weights_only=True)
                for img_id in tqdm(self.img_ids)
            ]
            self.embs = torch.stack(embs)
            embs_dict = {
                "ids": self.img_ids,
                "embs": self.embs,
            }
            torch.save(embs_dict, emb_pth)
        assert (
            len(self.embs) == len(self.img_ids)
        ), f"Number of embeddings {len(self.embs)} does not match number of images {len(self.img_ids)}"

    def get_target_img_ids(self, index) -> Dict[str, int]:
        """
        Returns the id of the target image and ground truth images for a given query

        Args:
            index (int): id of the query

        Returns:
             Dict[str, int]: dictionary containing target image id and a list of ground truth image ids
        """

        return {
            "target_img_id": self.annotations[index]["target_img_id"],
            "gt_img_ids": self.annotations[index]["gt_img_ids"],
        }

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index) -> dict:
        """
        Returns a specific item from the dataset based on the index.

        In 'relative' mode, the dataset yields dictionaries with the following keys:
            - [reference_img, reference_img_id, target_img, target_img_id, relative_caption, shared_concept, gt_img_ids,
            query_id] if split == val
            - [reference_img, reference_img_id, relative_caption, shared_concept, query_id]  if split == test
        """
        # Get the query id
        query_id = str(self.annotations[index]["id"])

        # Get relative caption and shared concept
        relative_caption = self.annotations[index]["relative_caption"]
        relative_caption = pre_caption(relative_caption, self.max_words)
        shared_concept = self.annotations[index]["shared_concept"]
        shared_concept = pre_caption(shared_concept, self.max_words)

        # Get the reference image
        reference_img_id = str(self.annotations[index]["reference_img_id"])
        reference_img_path = self.img_paths[self.img_ids_indexes_map[reference_img_id]]
        reference_img = Image.open(reference_img_path).convert("RGB")
        reference_img = self.transform(reference_img)

        if self.split == "test":
            return {
                "reference_img": reference_img,
                "reference_img_id": reference_img_id,
                "relative_caption": relative_caption,
                "shared_concept": shared_concept,
                "query_id": query_id,
            }

        # Get the target image and ground truth images
        target_img_id = str(self.annotations[index]["target_img_id"])
        gt_img_ids = [str(x) for x in self.annotations[index]["gt_img_ids"]]
        target_img_path = self.img_paths[self.img_ids_indexes_map[target_img_id]]
        target_img = Image.open(target_img_path).convert("RGB")
        target_img = self.transform(target_img)

        # Pad ground truth image IDs with zeros for collate_fn
        gt_img_ids += [""] * (self.max_num_gts - len(gt_img_ids))

        return {
            "reference_img": reference_img,
            "reference_img_id": reference_img_id,
            "target_img": target_img,
            "target_img_id": target_img_id,
            "relative_caption": relative_caption,
            "shared_concept": shared_concept,
            "gt_img_ids": gt_img_ids,
            "query_id": query_id,
        }
