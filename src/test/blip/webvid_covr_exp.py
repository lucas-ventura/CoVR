import datetime
import time
from pathlib import Path

import einops
import torch
import torch.nn.functional as F

from src.test.blip.webvid_covr import eval_recall
from src.tools.files import json_dump


class TestWebVidCoVRTextOnly:
    def __init__(self, remove_self_similarity=True):
        self.remove_self_similarity = remove_self_similarity

    @torch.no_grad()
    def __call__(self, model, data_loader, fabric):
        model.eval()

        fabric.print("Computing features for evaluation...")
        start_time = time.time()

        tar_img_feats = []
        query_feats = []
        captions = []
        pair_ids = []

        for batch in data_loader:
            tar_feat = batch["tar_img_feat"]
            pair_id = batch["pair_id"]
            caption = batch["edit"]

            pair_ids.extend(pair_id.cpu().numpy().tolist())
            captions.extend(caption)

            device = pair_id.device

            text = model.tokenizer(
                caption,
                padding="longest",
                truncation=True,
                max_length=64,
                return_tensors="pt",
            ).to(device)

            # Shift encoder
            query_embs = model.text_encoder(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
                mode="text",
            )
            query_feat = query_embs.last_hidden_state[:, 0, :]
            query_feat = F.normalize(model.text_proj(query_feat), dim=-1)
            query_feats.append(query_feat.cpu())

            # Encode the target image
            tar_img_feats.append(tar_feat.cpu())

        query_feats = torch.cat(query_feats, dim=0)
        tar_img_feats = torch.cat(tar_img_feats, dim=0)

        query_feats = F.normalize(query_feats, dim=-1)
        tar_img_feats = F.normalize(tar_img_feats, dim=-1)

        ref_img_ids = [data_loader.dataset.pairid2ref[pair_id] for pair_id in pair_ids]
        tar_img_ids = [data_loader.dataset.pairid2tar[pair_id] for pair_id in pair_ids]

        ref_img_ids = torch.tensor(ref_img_ids, dtype=torch.long)
        tar_img_ids = torch.tensor(tar_img_ids, dtype=torch.long)

        if fabric.world_size > 1:
            # Gather tensors from every process
            query_feats = fabric.all_gather(query_feats)
            tar_img_feats = fabric.all_gather(tar_img_feats)
            ref_img_ids = fabric.all_gather(ref_img_ids)
            tar_img_ids = fabric.all_gather(tar_img_ids)

            query_feats = einops.rearrange(query_feats, "d b e -> (d b) e")
            tar_img_feats = einops.rearrange(tar_img_feats, "d b e -> (d b) e")
            ref_img_ids = einops.rearrange(ref_img_ids, "d b -> (d b)")
            tar_img_ids = einops.rearrange(tar_img_ids, "d b -> (d b)")

        if fabric.global_rank == 0:
            sim_q2t = (query_feats @ tar_img_feats.t()).cpu().numpy()

            if self.remove_self_similarity:
                for i in range(len(ref_img_ids)):
                    for j in range(len(tar_img_ids)):
                        if ref_img_ids[i] == tar_img_ids[j]:
                            sim_q2t[i][j] = -10

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print("Evaluation time {}".format(total_time_str))

            recalls = eval_recall(sim_q2t)
            recalls["annotation"] = Path(data_loader.dataset.annotation_pth).name
            fabric.print(recalls)

            # Save results
            self_sim = "" if self.remove_self_similarity else "_ss"
            json_dump(recalls, f"recalls_covr_txt{self_sim}.json")

            print(f"Recalls saved in {Path.cwd()} as recalls_covr_txt{self_sim}.json")

        fabric.barrier()


class TestWebVidCoVRVisualOnly:
    def __init__(self):
        pass

    @staticmethod
    @torch.no_grad()
    def __call__(model, data_loader, fabric):
        model.eval()

        fabric.print("Computing features for evaluation...")
        start_time = time.time()

        tar_img_feats = []
        query_feats = []
        pair_ids = []

        for batch in data_loader:
            ref_img = batch["ref_img"]
            tar_feat = batch["tar_img_feat"]
            pair_id = batch["pair_id"]
            pair_ids.extend(pair_id.cpu().numpy().tolist())

            ref_img_embs = model.visual_encoder(ref_img)
            query_feat = F.normalize(model.vision_proj(ref_img_embs[:, 0, :]), dim=-1)
            query_feats.append(query_feat.cpu())

            # Encode the target image
            tar_img_feats.append(tar_feat.cpu())

        query_feats = torch.cat(query_feats, dim=0)
        tar_img_feats = torch.cat(tar_img_feats, dim=0)

        query_feats = F.normalize(query_feats, dim=-1)
        tar_img_feats = F.normalize(tar_img_feats, dim=-1)

        ref_img_ids = [data_loader.dataset.pairid2ref[pair_id] for pair_id in pair_ids]
        tar_img_ids = [data_loader.dataset.pairid2tar[pair_id] for pair_id in pair_ids]

        ref_img_ids = torch.tensor(ref_img_ids, dtype=torch.long)
        tar_img_ids = torch.tensor(tar_img_ids, dtype=torch.long)

        if fabric.world_size > 1:
            # Gather tensors from every process
            query_feats = fabric.all_gather(query_feats)
            tar_img_feats = fabric.all_gather(tar_img_feats)
            ref_img_ids = fabric.all_gather(ref_img_ids)
            tar_img_ids = fabric.all_gather(tar_img_ids)

            query_feats = einops.rearrange(query_feats, "d b e -> (d b) e")
            tar_img_feats = einops.rearrange(tar_img_feats, "d b e -> (d b) e")
            ref_img_ids = einops.rearrange(ref_img_ids, "d b -> (d b)")
            tar_img_ids = einops.rearrange(tar_img_ids, "d b -> (d b)")

        if fabric.global_rank == 0:
            sim_q2t = (query_feats @ tar_img_feats.t()).cpu().numpy()

            # Add zeros where ref_img_id == tar_img_id
            for i in range(len(ref_img_ids)):
                for j in range(len(tar_img_ids)):
                    if ref_img_ids[i] == tar_img_ids[j]:
                        sim_q2t[i][j] = -10

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print("Evaluation time {}".format(total_time_str))

            recalls = eval_recall(sim_q2t)
            fabric.print(recalls)

            # Save results
            json_dump(recalls, "recalls_covr.json")

            print(f"Recalls saved in {Path.cwd()} as recalls_covr.json")

        fabric.barrier()
