import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sys.path.append(project_root)

from lavis.models import load_model_and_preprocess

from src.data.embs import ImageDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def main(args):
    dataset = ImageDataset(
        image_dir=args.image_dir,
        save_dir=args.save_dir,
        todo_ids=args.todo_ids,
        image_size=args.image_size,
    )

    print("Creating model")

    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_feature_extractor",
        model_type=args.model_type,
        is_eval=True,
        device=device,
    )
    dataset.transform = vis_processors["eval"]
    dataset.image_size = args.image_size

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    for imgs, video_ids in tqdm(loader):
        imgs = imgs.to(device)
        img_embs = model.extract_features({"image": imgs}, mode="image")
        img_feats = img_embs.image_embeds_proj.cpu()

        for img_feat, video_id in zip(img_feats, video_ids):
            torch.save(img_feat, args.save_dir / f"{video_id}.pth")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_dir", type=Path, required=True, help="Path to image directory"
    )
    parser.add_argument("--save_dir", type=Path)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--model_type", type=str, default="coco", choices=["coco", "pretrain_vitL"]
    )
    parser.add_argument("--todo_ids", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=364, choices=[224, 364])
    args = parser.parse_args()

    args.save_dir.mkdir(exist_ok=True)

    main(args)
