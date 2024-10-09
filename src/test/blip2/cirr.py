import datetime
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.tools.files import json_dump
from src.tools.utils import concat_all_gather


class TestCirr:
    def __init__(self):
        pass

    @staticmethod
    @torch.no_grad()
    def __call__(model, data_loader, fabric):
        model.eval()

        fabric.print("Computing features for test...")
        start_time = time.time()

        vl_feats = []
        pair_ids = []
        for batch in data_loader:
            ref_img = batch["ref_img"]
            caption = batch["edit"]
            pair_id = batch["pair_id"]

            pair_ids.extend(pair_id.cpu().numpy().tolist())

            device = ref_img.device

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

            output = model.Qformer.bert(
                text_tokens.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=ref_img_embs,
                encoder_attention_mask=ref_img_atts,
                return_dict=True,
            )

            vl_embs = output.last_hidden_state[:, : query_tokens.size(1), :]
            vl_feat = F.normalize(model.text_proj(vl_embs), dim=-1)
            vl_feats.append(vl_feat.cpu())

        pair_ids = torch.tensor(pair_ids, dtype=torch.long)
        vl_feats = torch.cat(vl_feats, dim=0)

        vl_feats = concat_all_gather(vl_feats, fabric)
        pair_ids = concat_all_gather(pair_ids, fabric)

        if fabric.global_rank == 0:
            pair_ids = pair_ids.cpu().numpy().tolist()

            assert len(vl_feats) == len(pair_ids)
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
            vl_feats = vl_feats.to("cpu")

            # sims_q2t = torch.einsum("iqe,jke->ijqk", vl_feats, tar_feats)
            # Process in batches to avoid memory issues
            batch_size = 100
            sims_q2t = []
            for i in range(0, vl_feats.size(0), batch_size):
                vl_feats_batch = vl_feats[i : i + batch_size]
                sim_batch = torch.einsum("iqe,jke->ijqk", vl_feats_batch, tar_feats)
                sims_q2t.append(sim_batch)
            sims_q2t = torch.cat(sims_q2t, dim=0)

            sims_q2t = sims_q2t.max(dim=-1)[0]
            sims_q2t = sims_q2t.max(dim=-1)[0]

            # Create a mapping from pair_id to row index for faster lookup
            pairid2index = {pair_id: i for i, pair_id in enumerate(pair_ids)}

            # Create a mapping from target_id to column index for faster lookup
            tarid2index = {tar_id: j for j, tar_id in enumerate(id2emb.keys())}

            # Update the similarity matrix based on the condition
            for pair_id in pair_ids:
                que_id = data_loader.dataset.pairid2ref[pair_id]
                if que_id in tarid2index:
                    sims_q2t[pairid2index[pair_id], tarid2index[que_id]] = -100
            sims_q2t = sims_q2t.cpu().numpy()

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print("Evaluation time {}".format(total_time_str))

            recalls = {}
            recalls["version"] = "rc2"
            recalls["metric"] = "recall"

            recalls_subset = {}
            recalls_subset["version"] = "rc2"
            recalls_subset["metric"] = "recall_subset"

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

            json_dump(recalls, "recalls_cirr.json")
            json_dump(recalls_subset, "recalls_cirr_subset.json")

            print(f"Recalls saved in {Path.cwd()}/recalls_cirr.json")

        fabric.barrier()
