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

from src.data.embs import VideoDataset
from src.model.blip_embs import blip_embs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_blip_config(model="base"):
    config = dict()
    if model == "base":
        config["pretrained"] = (
            "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth"
        )
        # ] = "/linkhome/rech/genuvt01/ucp99db/.cache/torch/hub/checkpoints/model_base_retrieval_coco.pth"

        config["vit"] = "base"
        config["batch_size_train"] = 32
        config["batch_size_test"] = 16
        config["vit_grad_ckpt"] = True
        config["vit_ckpt_layer"] = 4
        config["init_lr"] = 1e-5
    elif model == "large":
        config["pretrained"] = (
            "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_retrieval_coco.pth"
        )
        config["vit"] = "large"
        config["batch_size_train"] = 16
        config["batch_size_test"] = 32
        config["vit_grad_ckpt"] = True
        config["vit_ckpt_layer"] = 12
        config["init_lr"] = 5e-6

    config["image_size"] = 384
    config["queue_size"] = 57600
    config["alpha"] = 0.4
    config["k_test"] = 256
    config["negative_all_rank"] = True

    return config


@torch.no_grad()
def main(args):
    save_tokens = "tokens-" if args.save_all_tokens else ""
    save_dir = (
        args.video_dir.parent / f"blip-vid-embs-{save_tokens}{args.model_type}-all"
    )
    save_dir.mkdir(exist_ok=True)

    dataset = VideoDataset(
        video_dir=args.video_dir,
        todo_ids=args.todo_ids,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        frames_video=args.frames_video,
        save_dir=save_dir,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    print(f"Creating model {args.model_type}")
    config = get_blip_config(args.model_type)
    model = blip_embs(
        pretrained=config["pretrained"],
        image_size=config["image_size"],
        vit=config["vit"],
        vit_grad_ckpt=config["vit_grad_ckpt"],
        vit_ckpt_layer=config["vit_ckpt_layer"],
        queue_size=config["queue_size"],
        negative_all_rank=config["negative_all_rank"],
    )

    model = model.to(device)
    model.eval()

    for video_ids, f_idxs, frames in tqdm(loader):
        frames = frames.to(device)
        bs, nf, c, h, w = frames.shape
        frames = frames.view(bs * nf, c, h, w)
        frm_embs = model.visual_encoder(frames)
        if args.save_all_tokens:
            frm_feats = frm_embs.cpu()
            frm_feats = frm_feats.view(bs, nf, 577, 1024)
        else:
            frm_feats = F.normalize(model.vision_proj(frm_embs[:, 0, :]), dim=-1).cpu()
            frm_feats = frm_feats.view(bs, nf, -1)

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
    parser.add_argument("--video_dir", type=Path, default="datasets/WebVid/8M/train/")
    parser.add_argument("--todo_ids", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--model_type", type=str, default="large", choices=["base", "large"]
    )
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--frames_video", type=int, default=15)
    parser.add_argument("--save_all_tokens", action="store_true")

    args = parser.parse_args()

    assert args.video_dir.exists(), f"{args.video_dir} does not exist"

    main(args)
