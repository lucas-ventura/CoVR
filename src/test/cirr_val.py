import datetime
import time

import einops
import numpy as np
import torch
import torch.nn.functional as F


class TestEvaluate:
    def __init__(self):
        pass

    @staticmethod
    @torch.no_grad()
    def __call__(model, data_loader, fabric):
        evaluate(model, data_loader, fabric)


@torch.no_grad()
def evaluate(model, data_loader, fabric):
    model.eval()

    fabric.print("Computing features for evaluation...")
    start_time = time.time()

    tar_img_feats = []
    query_feats = []
    captions = []
    pair_ids = []

    for ref_img, tar_feat, caption, pair_id, *_ in data_loader:
        pair_ids.extend(pair_id.cpu().numpy().tolist())
        captions.extend(caption)

        device = ref_img.device

        ref_img_embs = model.visual_encoder(ref_img)
        ref_img_atts = torch.ones(ref_img_embs.size()[:-1], dtype=torch.long).to(device)

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

    if fabric.world_size > 1:
        # Gather tensors from every process
        query_feats = fabric.all_gather(query_feats)
        tar_img_feats = fabric.all_gather(tar_img_feats)

        query_feats = einops.rearrange(query_feats, "d b e -> (d b) e")
        tar_img_feats = einops.rearrange(tar_img_feats, "d b e -> (d b) e")

    if fabric.global_rank == 0:
        ref_img_ids = np.array(
            [data_loader.dataset.pairid2ref[pair_id] for pair_id in pair_ids]
        )
        tar_img_ids = np.array(
            [data_loader.dataset.pairid2tar[pair_id] for pair_id in pair_ids]
        )

        sim_q2t = (query_feats @ tar_img_feats.t()).cpu().numpy()

        # Add zeros where ref_img_id == tar_img_id
        for i in range(len(ref_img_ids)):
            for j in range(len(tar_img_ids)):
                if ref_img_ids[i] == tar_img_ids[j]:
                    sim_q2t[i][j] = -10

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Evaluation time {}".format(total_time_str))

        eval_result = eval_recall(sim_q2t)
        fabric.print(eval_result)

        # Compute Recall_subset@K
        # Code adapted from https://github.com/Cuberick-Orion/CIRPLANT/blob/cd5798dbeb1902aa9b95c323c2f90eb90a757918/model/OSCAR/OSCAR_CIRPLANT.py#L318
        recalls_subset = {}
        assert len(sim_q2t) == len(pair_ids)
        for pair_id, query_sims in zip(pair_ids, sim_q2t):
            sorted_indices = np.argsort(query_sims)[::-1]

            members = data_loader.dataset.pairid2members[pair_id]
            query_id_recalls_subset = [
                target for target in tar_img_ids[sorted_indices] if target in members
            ][:3]
            assert (
                len(query_id_recalls_subset) > 0
            ), f"pair_id: {pair_id} has no recalls"
            recalls_subset[str(pair_id)] = query_id_recalls_subset

        all_target_captions_soft = {
            ann["pairid"]: ann["target_soft"] for ann in data_loader.dataset.annotation
        }
        for k in [1, 2, 3]:
            r = 0
            for pair_id, query_id_recalls_subset in recalls_subset.items():
                highest_r = 0.0
                for ii, ss in all_target_captions_soft[int(pair_id)].items():
                    if ii in query_id_recalls_subset[:k]:
                        highest_r = max(highest_r, ss)
                r += highest_r
            r /= len(recalls_subset)
            fabric.print(f"Recall_subset@{k}: {r*100:.2f}")

    fabric.barrier()


@torch.no_grad()
def eval_recall(scores_q2t):
    # Query->Target
    ranks = np.zeros(scores_q2t.shape[0])

    for index, score in enumerate(scores_q2t):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == index)[0][0]

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    tr50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)

    tr_mean = (tr1 + tr5 + tr10) / 3

    eval_result = {
        "R1": round(tr1, 4),
        "R5": round(tr5, 4),
        "R10": round(tr10, 4),
        "R50": round(tr50, 4),
        "R_mean": round(tr_mean, 4),
    }
    return eval_result
