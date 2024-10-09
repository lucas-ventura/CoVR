import datetime
import time
from collections import OrderedDict

import einops
import numpy as np
import torch
import torch.nn.functional as F


class ValCirr:
    def __init__(self):
        pass

    @staticmethod
    @torch.no_grad()
    def __call__(model, data_loader, fabric):
        model.eval()

        fabric.print("Computing features for validation...")
        start_time = time.time()

        query_feats = []
        pair_ids = []
        for batch in data_loader:
            ref_img = batch["ref_img"]
            caption = batch["edit"]
            pair_id = batch["pair_id"]

            pair_ids.extend(pair_id.cpu().numpy().tolist())

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

        pair_ids = torch.tensor(pair_ids, dtype=torch.long)
        query_feats = torch.cat(query_feats, dim=0)

        if fabric.world_size > 1:
            # Gather tensors from every process
            query_feats = fabric.all_gather(query_feats)
            pair_ids = fabric.all_gather(pair_ids)

            query_feats = einops.rearrange(query_feats, "d b e -> (d b) e")
            pair_ids = einops.rearrange(pair_ids, "d b -> (d b)")

        if fabric.global_rank == 0:
            pair_ids = pair_ids.cpu().numpy().tolist()

            assert len(query_feats) == len(pair_ids)
            img_ids = [data_loader.dataset.pairid2ref[pair_id] for pair_id in pair_ids]
            assert len(img_ids) == len(pair_ids)

            id2emb = OrderedDict()
            for img_id, target_emb_pth in data_loader.dataset.id2embpth.items():
                if img_id not in id2emb:
                    tar_emb = F.normalize(
                        torch.load(target_emb_pth, weights_only=True).cpu(), dim=-1
                    )
                    id2emb[img_id] = tar_emb

            tar_feats = torch.stack(list(id2emb.values()), dim=0).to("cpu")
            query_feats = query_feats.to("cpu")
            sims_q2t = query_feats @ tar_feats.T

            assert sims_q2t.shape == (
                4181,
                8102,
            ), f"Expected (4181, 8102), got {sims_q2t.shape}"

            # Create a mapping from pair_id to row index for faster lookup
            pairid2index = {pair_id: i for i, pair_id in enumerate(pair_ids)}

            # Create a mapping from target_id to column index for faster lookup
            tarid2index = {tar_id: j for j, tar_id in enumerate(id2emb.keys())}

            # Update the similarity matrix based on the condition
            for pair_id, query_feat in zip(pair_ids, query_feats):
                que_id = data_loader.dataset.pairid2ref[pair_id]
                if que_id in tarid2index:
                    sims_q2t[pairid2index[pair_id], tarid2index[que_id]] = -100
            sims_q2t = sims_q2t.cpu().numpy()

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print("Evaluation time {}".format(total_time_str))

            recalls = {}
            recalls_subset = {}
            target_imgs = np.array(list(id2emb.keys()))

            assert len(sims_q2t) == len(pair_ids)
            for pair_id, query_sims in zip(pair_ids, sims_q2t):
                sorted_indices = np.argsort(query_sims)[::-1]

                query_id_recalls = list(target_imgs[sorted_indices][:50])
                recalls[str(pair_id)] = query_id_recalls

                members = data_loader.dataset.pairid2members[pair_id]
                query_id_recalls_subset = [
                    target
                    for target in target_imgs[sorted_indices]
                    if target in members
                ][:3]
                recalls_subset[str(pair_id)] = query_id_recalls_subset

            # Compute Recall@K
            paird2target = {
                ann["pairid"]: ann["target_hard"]
                for ann in data_loader.dataset.annotation
            }
            for k in [1, 5, 10, 50]:
                r = 0
                for pair_id, query_id_recalls in recalls.items():
                    r += paird2target[int(pair_id)] in query_id_recalls[:k]
                r /= len(recalls)
                fabric.print(f"Recall@{k}: {r*100:.2f}")

            # Compute Recall_subset@K
            paird2target_soft = {
                ann["pairid"]: ann["target_soft"]
                for ann in data_loader.dataset.annotation
            }
            for k in [1, 2, 3]:
                r = 0
                for pair_id, query_id_recalls_subset in recalls_subset.items():
                    highest_r = 0.0
                    for ii, ss in paird2target_soft[int(pair_id)].items():
                        if ii in query_id_recalls_subset[:k]:
                            highest_r = max(highest_r, ss)
                    r += highest_r
                r /= len(recalls_subset)
                fabric.print(f"Recall_subset@{k}: {r*100:.2f}")

        fabric.barrier()
