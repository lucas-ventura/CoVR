import datetime
import time
from pathlib import Path

import einops
import numpy as np
import torch
import torch.nn.functional as F

from src.tools.files import json_dump


class TestWebVidCoVR:
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
            ref_img = batch["ref_img"]
            tar_feat = batch["tar_img_feat"]
            caption = batch["edit"]
            pair_id = batch["pair_id"]
            pair_ids.extend(pair_id.cpu().numpy().tolist())
            captions.extend(caption)

            device = ref_img.device

            ref_img_embs = model.visual_encoder(ref_img)
            ref_img_atts = torch.ones(ref_img_embs.size()[:-1], dtype=torch.long).to(
                device
            )

            text = model.tokenizer(
                caption,
                padding="longest",
                truncation=True,
                max_length=64,
                return_tensors="pt",
            ).to(device)

            # Shift encoder
            encoder_input_ids = text.input_ids.clone()
            encoder_input_ids[:, 0] = model.tokenizer.enc_token_id
            query_embs = model.text_encoder(
                encoder_input_ids,
                attention_mask=text.attention_mask,
                encoder_hidden_states=ref_img_embs,
                encoder_attention_mask=ref_img_atts,
                return_dict=True,
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
            json_dump(recalls, f"recalls_covr{self_sim}.json")

            print(f"Recalls saved in {Path.cwd()} as recalls_covr{self_sim}.json")

        fabric.barrier()


@torch.no_grad()
def eval_recall(scores_q2t):
    # Query->Target
    ranks = np.zeros(scores_q2t.shape[0])

    for index, score in enumerate(scores_q2t):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == index)[0][0]

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)  # type: ignore
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    tr50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)

    tr_mean3 = (tr1 + tr5 + tr10) / 3
    tr_mean4 = (tr1 + tr5 + tr10 + tr50) / 4

    eval_result = {
        "R1": round(tr1, 2),
        "R5": round(tr5, 2),
        "R10": round(tr10, 2),
        "R50": round(tr50, 2),
        "meanR3": round(tr_mean3, 2),
        "meanR4": round(tr_mean4, 2),
    }
    return eval_result
