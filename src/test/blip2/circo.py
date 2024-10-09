import datetime
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import einops
import numpy as np
import torch
import torch.nn.functional as F

from src.tools.files import json_dump


class TestCirco:
    def __init__(self, split="test"):
        assert split in ["val", "test"]
        self.split = split

    @torch.no_grad()
    def __call__(self, model, data_loader, fabric):
        model.eval()

        fabric.print("Computing features for test...")
        start_time = time.time()

        query_feats = []
        query_ids = []
        ref_img_ids = []
        for batch in data_loader:
            ref_img = batch["reference_img"]
            caption = batch["relative_caption"]
            device = ref_img.device

            query_ids.extend(batch["query_id"])
            ref_img_ids.extend(batch["reference_img_id"])

            ref_img_embs = model.ln_vision(model.visual_encoder(ref_img))
            ref_img_atts = torch.ones(ref_img_embs.size()[:-1], dtype=torch.long).to(
                device
            )

            text_tokens = model.tokenizer(
                caption,
                padding="longest",
                truncation=True,
                max_length=64,
                return_tensors="pt",
            ).to(device)

            # Shift encoder
            query_tokens = model.query_tokens.expand(ref_img_embs.shape[0], -1, -1)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                device
            )
            attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)

            query_embs = model.Qformer.bert(
                text_tokens.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=ref_img_embs,
                encoder_attention_mask=ref_img_atts,
                return_dict=True,
            )

            query_feat = query_embs.last_hidden_state[:, : query_tokens.size(1), :]
            query_feat = F.normalize(model.text_proj(query_feat), dim=-1)
            query_feats.append(query_feat.cpu())

        query_feats = torch.cat(query_feats, dim=0)
        ref_img_ids = torch.tensor([int(id) for id in ref_img_ids], dtype=torch.long)
        query_ids = torch.tensor([int(id) for id in query_ids], dtype=torch.long)

        tar_img_feats = data_loader.dataset.embs.to(query_feats.device)
        tar_ids = torch.Tensor(data_loader.dataset.img_ids).long()

        if fabric.world_size > 1:
            # Gather tensors from every process
            query_feats = fabric.all_gather(query_feats)
            ref_img_ids = fabric.all_gather(ref_img_ids)
            query_ids = fabric.all_gather(query_ids)

            query_feats = einops.rearrange(query_feats, "d b q e -> (d b) q e")
            ref_img_ids = einops.rearrange(ref_img_ids, "d b -> (d b)")
            query_ids = einops.rearrange(query_ids, "d b -> (d b)")

        if fabric.global_rank == 0:
            ref_img_ids = ref_img_ids.cpu().numpy().tolist()

            assert len(ref_img_ids) == len(query_feats)
            assert len(ref_img_ids) == len(query_ids)

            query_feats = query_feats.cpu()
            tar_img_feats = tar_img_feats.cpu()

            query_feats = query_feats.mean(dim=1)
            tar_img_feats = tar_img_feats.mean(dim=1)

            sims_q2t = query_feats @ tar_img_feats.T

            # Set the similarity scores to -100 where query_id == tar_id
            ref_ids = torch.Tensor(ref_img_ids).long()
            mask = (ref_ids[:, None] == tar_ids[None, :]).float()
            sims_q2t -= 100 * mask
            assert (sims_q2t < -10).sum().item() == len(
                ref_ids
            ), "Not all ref_ids are in the target set"
            sims_q2t = sims_q2t.cpu().numpy()
            tar_ids = tar_ids.cpu().numpy()

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print(f"Evaluation time {total_time_str}")

            recalls = {}
            assert len(sims_q2t) == len(ref_img_ids)
            assert len(sims_q2t) == len(query_ids)
            for query_id, query_sims in zip(query_ids, sims_q2t):
                sorted_indices = np.argsort(query_sims)[::-1]

                query_id_recalls = list(tar_ids[sorted_indices][:50])
                query_id_recalls = [int(id) for id in query_id_recalls]
                recalls[str(query_id.item())] = query_id_recalls

            if self.split == "test":
                json_dump(recalls, "recalls_circo-test.json")
                print(f"Recalls saved in {Path.cwd()}/recalls_circo-test.json")

            elif self.split == "val":
                ap_atk, recall_atk = compute_metrics(data_loader.dataset, recalls)
                recalls = {}
                for k, v in ap_atk.items():
                    recalls[f"mAP@{k}"] = v
                    print(f"mAP@{k}: {v:.2f}")

                for k, v in recall_atk.items():
                    recalls[f"Recall@{k}"] = v
                    print(f"Recall@{k}: {v:.2f}")

                json_dump(recalls, "recalls_circo-val.json")

                print(f"Recalls saved in {Path.cwd()}/recalls_circo-val.json")

        fabric.barrier()


def compute_metrics(
    dataset, predictions_dict: Dict[int, List[int]], ranks: List[int] = [5, 10, 25, 50]
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Computes the Average Precision (AP) and Recall for a given set of predictions.

    Args:
        data_path (Path): Path where the CIRCO datasset is located
        predictions_dict (Dict[int, List[int]]): Predictions of image ids for each query id
        ranks (List[int]): Ranks to consider in the evaluation (e.g., [5, 10, 20])

    Returns:
        Tuple[Dict[int, float], Dict[int, float]]: Dictionaries with the AP and Recall for each rank
    """

    # Initialize empty dictionaries to store the AP and Recall values for each rank
    aps_atk = defaultdict(list)
    recalls_atk = defaultdict(list)

    # Iterate through each query id and its corresponding predictions
    for query_id, predictions in predictions_dict.items():
        target = dataset.get_target_img_ids(int(query_id))
        gt_img_ids = target["gt_img_ids"]
        target_img_id = target["target_img_id"]

        # gt_img_ids = np.trim_zeros(gt_img_ids)  # remove trailing zeros added for collate_fn (when using dataloader)

        predictions = np.array(predictions, dtype=int)
        ap_labels = np.isin(predictions, gt_img_ids)
        precisions = (
            np.cumsum(ap_labels, axis=0) * ap_labels
        )  # Consider only positions corresponding to GTs
        precisions = precisions / np.arange(
            1, ap_labels.shape[0] + 1
        )  # Compute precision for each position

        # Compute the AP and Recall for the given ranks
        for rank in ranks:
            aps_atk[rank].append(
                float(np.sum(precisions[:rank]) / min(len(gt_img_ids), rank))
            )

        recall_labels = predictions == target_img_id
        for rank in ranks:
            recalls_atk[rank].append(float(np.sum(recall_labels[:rank])))

    # Compute the mean AP and Recall for each rank and store them in a dictionary
    ap_atk = {}
    recall_atk = {}
    for rank in ranks:
        ap_atk[rank] = round(float(np.mean(aps_atk[rank])) * 100, 2)
        recall_atk[rank] = round(float(np.mean(recalls_atk[rank])) * 100, 2)
    return ap_atk, recall_atk
