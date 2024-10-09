import datetime
import time
from pathlib import Path
from typing import Dict, List

import einops
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tabulate import tabulate

from src.tools.files import json_dump, json_load


class TestFashionIQ:
    def __init__(self, category: str):
        self.category = category
        pass

    @torch.no_grad()
    def __call__(self, model, data_loader, fabric):
        model.eval()

        fabric.print("Computing features for evaluation...")
        start_time = time.time()

        query_feats = []
        captions = []
        idxs = []
        for batch in data_loader:
            ref_img = batch["ref_img"]
            caption = batch["edit"]
            idx = batch["pair_id"]

            idxs.extend(idx.cpu().numpy().tolist())
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

        query_feats = torch.cat(query_feats, dim=0)
        query_feats = F.normalize(query_feats, dim=-1)
        idxs = torch.tensor(idxs, dtype=torch.long)

        if fabric.world_size > 1:
            # Gather tensors from every process
            query_feats = fabric.all_gather(query_feats)
            idxs = fabric.all_gather(idxs)

            query_feats = einops.rearrange(query_feats, "d b e -> (d b) e")
            idxs = einops.rearrange(idxs, "d b -> (d b)")

        if fabric.global_rank == 0:
            idxs = idxs.cpu().numpy()
            ref_img_ids = [data_loader.dataset.pairid2ref[idx] for idx in idxs]
            ref_img_ids = [data_loader.dataset.int2id[id] for id in ref_img_ids]

            tar_img_feats = []
            tar_img_ids = []
            for target_id in data_loader.dataset.target_ids:
                tar_img_ids.append(target_id)
                target_emb_pth = data_loader.dataset.id2embpth[target_id]
                target_feat = torch.load(target_emb_pth, weights_only=True).cpu()
                tar_img_feats.append(target_feat.cpu())
            tar_img_feats = torch.stack(tar_img_feats)
            tar_img_feats = F.normalize(tar_img_feats, dim=-1)
            tar_img_feats = tar_img_feats.to(query_feats.device)

            sim_q2t = (query_feats @ tar_img_feats.t()).cpu()

            # Add zeros where ref_img_id == tar_img_id
            for i in range(len(ref_img_ids)):
                for j in range(len(tar_img_ids)):
                    if ref_img_ids[i] == tar_img_ids[j]:
                        sim_q2t[i][j] = -10

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print("Evaluation time {}".format(total_time_str))

            ref_img_ids = np.array(ref_img_ids)
            tar_img_ids = np.array(tar_img_ids)

            cor_img_ids = [data_loader.dataset.pairid2tar[idx] for idx in idxs]
            cor_img_ids = [data_loader.dataset.int2id[id] for id in cor_img_ids]

            recalls = get_recalls_labels(sim_q2t, cor_img_ids, tar_img_ids)
            fabric.print(recalls)

            # Save results
            json_dump(recalls, f"recalls_fiq-{self.category}.json")

            print(f"Recalls saved in {Path.cwd()} as recalls_fiq-{self.category}.json")

            mean_results(fabric=fabric)

        fabric.barrier()


# From google-research/composed_image_retrieval
def recall_at_k_labels(sim, query_lbls, target_lbls, k=10):
    distances = 1 - sim
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(target_lbls)[sorted_indices]
    labels = torch.tensor(
        sorted_index_names
        == np.repeat(np.array(query_lbls), len(target_lbls)).reshape(
            len(query_lbls), -1
        )
    )
    assert torch.equal(
        torch.sum(labels, dim=-1).int(), torch.ones(len(query_lbls)).int()
    )
    return round((torch.sum(labels[:, :k]) / len(labels)).item() * 100, 2)


def get_recalls_labels(
    sims, query_lbls, target_lbls, ks: List[int] = [1, 5, 10, 50]
) -> Dict[str, float]:
    return {f"R{k}": recall_at_k_labels(sims, query_lbls, target_lbls, k) for k in ks}


def mean_results(dir=".", fabric=None, save=True):
    dir = Path(dir)
    recall_pths = list(dir.glob("recalls_fiq-*.json"))
    recall_pths.sort()
    if len(recall_pths) != 3:
        return

    df = {}
    for pth in recall_pths:
        name = pth.name.split("_")[1].split(".")[0]
        data = json_load(pth)
        df[name] = data

    df = pd.DataFrame(df)

    # FASHION-IQ
    df_fiq = df[df.columns[df.columns.str.contains("fiq")]]
    assert len(df_fiq.columns) == 3
    df_fiq["Average"] = df_fiq.mean(axis=1)
    df_fiq["Average"] = df_fiq["Average"].apply(lambda x: round(x, 2))

    headers = [
        "dress\nR10",
        "dress\nR50",
        "shirt\nR10",
        "shirt\nR50",
        "toptee\nR10",
        "toptee\nR50",
        "Average\nR10",
        "Average\nR50",
    ]
    fiq = []
    for category in ["fiq-dress", "fiq-shirt", "fiq-toptee", "Average"]:
        for recall in ["R10", "R50"]:
            value = df_fiq.loc[recall, category]
            value = str(value).zfill(2)
            fiq.extend([value])
    if fabric is None:
        print(tabulate([fiq], headers=headers, tablefmt="latex_raw"))
        print(" & ".join(fiq))
    else:
        fabric.print(tabulate([fiq], headers=headers))
        fabric.print(" & ".join(fiq))

    if save:
        df_mean = df_fiq["Average"].to_dict()
        df_mean = {k + "_mean": round(v, 2) for k, v in df_mean.items()}
        json_dump(df_mean, "recalls_fiq-mean.json")
