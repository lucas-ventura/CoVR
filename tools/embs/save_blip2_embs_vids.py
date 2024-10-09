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

from src.data.embs import VideoDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def main(args):
    if args.model_type == "coco":
        save_dir = args.video_dir.parent / f"blip2-vid-embs-large-all"
    else:
        save_dir = args.video_dir.parent / f"blip2-vid-embs-{args.model_type}-all"
    save_dir.mkdir(exist_ok=True)

    dataset = VideoDataset(
        video_dir=args.video_dir,
        todo_ids=args.todo_ids,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        frames_video=args.frames_video,
        save_dir=save_dir,
        image_size=args.image_size,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    print("Creating model")

    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_feature_extractor",
        model_type=args.model_type,
        is_eval=True,
        device=device,
    )
    dataset.transform = vis_processors["eval"]
    dataset.pixel_size = args.image_size

    for video_ids, f_idxs, frames in tqdm(loader):
        frames = frames.to(device)
        bs, nf, c, h, w = frames.shape
        frames = frames.view(bs * nf, c, h, w)

        frm_feats = model.extract_features({"image": frames}, mode="image")
        frm_feats = frm_feats.image_embeds_proj.view(bs, nf, 32, 256).cpu()

        for video_id, f_idx, frm_feat in zip(video_ids, f_idxs, frm_feats):
            # remove the features with f_idx=-1
            frm_feat = frm_feat[f_idx > -1]
            f_idx = f_idx[f_idx > -1]
            if len(f_idx) == 0:
                continue
            save_pth = save_dir / f"{video_id}.pth"
            if save_pth.exists():
                continue
            save_pth.parent.mkdir(exist_ok=True)

            torch.save(frm_feat, save_pth)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_dir", type=Path, required=True, help="Path to video directory"
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--model_type", type=str, default="coco", choices=["coco", "pretrain_vitL"]
    )
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--frames_video", type=int, default=15)
    parser.add_argument("--todo_ids", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=364, choices=[224, 364])
    args = parser.parse_args()

    main(args)
