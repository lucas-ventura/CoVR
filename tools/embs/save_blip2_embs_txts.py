import os
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sys.path.append(project_root)

from lavis.models import load_model_and_preprocess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextDataset(Dataset):
    def __init__(
        self,
        data_path,
        txt_processors,
        column="txt2",
    ):
        if data_path.suffix == ".csv":
            self.df = pd.read_csv(data_path)
            self.texts = list(set(self.df[column].unique().tolist()))
        elif data_path.suffix == ".txt":
            with open(data_path, "r") as f:
                texts = f.readlines()
            self.texts = [x.rstrip("\n") for x in set(texts)]
        elif data_path.suffix == ".json":
            import json

            with open(data_path, "r") as f:
                texts = json.load(f)

            self.texts = list(set([txt["caption"] for txt in texts]))
        self.texts.sort()
        self.txt_processors = txt_processors

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        txt = self.texts[index]
        txt = self.txt_processors["eval"](txt)

        return txt


@torch.no_grad()
def main(args):
    print("Creating model")
    model, _, txt_processors = load_model_and_preprocess(
        name="blip2_feature_extractor",
        model_type=args.model_type,
        is_eval=True,
        device=device,
    )

    dataset = TextDataset(
        args.data_path,
        txt_processors,
        column=args.column,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    text_feats = []
    for txts in tqdm(loader):
        txt_embs = model.extract_features({"text_input": txts}, mode="text")
        text_feat = txt_embs.text_embeds_proj[:, 0, :].cpu()
        text_feats.append(text_feat)

    text_feats = torch.cat(text_feats, dim=0)
    save_obj = {
        "texts": dataset.texts,
        "feats": text_feats,
    }
    torch.save(save_obj, args.save_dir / f"{args.column}_{args.data_path.stem}.pth")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=Path)
    parser.add_argument("save_dir", type=Path)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--model_type", type=str, default="coco", choices=["coco", "pretrain_vitL"]
    )
    parser.add_argument("--column", type=str, default="txt2")
    args = parser.parse_args()

    args.save_dir.mkdir(exist_ok=True)

    main(args)
