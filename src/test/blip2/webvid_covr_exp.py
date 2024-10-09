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

            text_tokens = model.tokenizer(
                caption,
                padding="longest",
                truncation=True,
                max_length=64,
                return_tensors="pt",
            ).to(device)

            # Shift encoder
            query_embs = model.Qformer.bert(
                text_tokens.input_ids,
                attention_mask=text_tokens.attention_mask,
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
            tar_img_feats = einops.rearrange(tar_img_feats, "d b q e -> (d b) q e")
            ref_img_ids = einops.rearrange(ref_img_ids, "d b -> (d b)")
            tar_img_ids = einops.rearrange(tar_img_ids, "d b -> (d b)")

        if fabric.global_rank == 0:
            tar_img_feats = tar_img_feats.mean(dim=1)
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

            print(f"Recalls saved in {Path.cwd()}/recalls_covr_txt{self_sim}.json")

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
            device = ref_img.device

            ref_img_embs = model.ln_vision(model.visual_encoder(ref_img))

            image_atts = torch.ones(ref_img_embs.size()[:-1], dtype=torch.long).to(
                device
            )

            query_tokens = model.query_tokens.expand(ref_img_embs.shape[0], -1, -1)

            query_output = model.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=ref_img_embs,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            image_embeds = query_output.last_hidden_state
            query_feat = F.normalize(model.vision_proj(image_embeds), dim=-1)

            # query_feat = F.normalize(model.vision_proj(ref_img_embs[:, 0, :]), dim=-1)
            query_feat = query_feat.mean(dim=1)
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
            tar_img_feats = tar_img_feats.mean(dim=1)
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
            json_dump(recalls, "recalls_covr_vis.json")

            print(f"Recalls saved in {Path.cwd()}/recalls_covr_vis.json")

        fabric.barrier()


class TestWebVidCoVRAvg:
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
            pair_id = batch["pair_id"]
            caption = batch["edit"]

            pair_ids.extend(pair_id.cpu().numpy().tolist())
            captions.extend(caption)

            device = pair_id.device

            # Text
            text_tokens = model.tokenizer(
                caption,
                padding="longest",
                truncation=True,
                max_length=64,
                return_tensors="pt",
            ).to(device)

            query_embs = model.Qformer.bert(
                text_tokens.input_ids,
                attention_mask=text_tokens.attention_mask,
                return_dict=True,
            )
            query_txt_feat = query_embs.last_hidden_state[:, 0, :]
            query_txt_feat = F.normalize(model.text_proj(query_txt_feat), dim=-1)

            # Image
            ref_img_embs = model.ln_vision(model.visual_encoder(ref_img))

            image_atts = torch.ones(ref_img_embs.size()[:-1], dtype=torch.long).to(
                device
            )

            query_tokens = model.query_tokens.expand(ref_img_embs.shape[0], -1, -1)

            query_output = model.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=ref_img_embs,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            image_embeds = query_output.last_hidden_state
            query_img_feat = F.normalize(model.vision_proj(image_embeds), dim=-1)
            query_img_feat = query_img_feat.mean(dim=1)

            # Average
            query_feat = (query_txt_feat + query_img_feat) / 2
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
            tar_img_feats = tar_img_feats.mean(dim=1)
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
            json_dump(recalls, f"recalls_covr_avg{self_sim}.json")

            print(f"Recalls saved in {Path.cwd()}/recalls_covr_avg{self_sim}.json")

        fabric.barrier()


class TestWebVidCoVRCaptions:
    def __init__(
        self,
        w_txt: float = 0.5,
        w_img: float = 0.5,
        remove_self_similarity: bool = True,
        dataset: str = "covr_blip2-captions",
    ):
        self.remove_self_similarity = remove_self_similarity
        self.dataset = dataset
        self.w_txt = w_txt / (w_txt + w_img)
        self.w_img = w_img / (w_txt + w_img)

    @torch.no_grad()
    def __call__(self, model, data_loader, fabric):
        model.eval()

        fabric.print("Computing features for evaluation...")
        start_time = time.time()

        tar_img_feats = []
        tar_txt_feats = []
        query_feats = []
        captions = []
        pair_ids = []

        for batch in data_loader:
            ref_img = batch["ref_img"]
            tar_img_feat = batch["tar_img_feat"]
            tar_txt_feat = batch["tar_txt_feat"]
            caption = batch["edit"]
            pair_id = batch["pair_id"]

            pair_ids.extend(pair_id.cpu().numpy().tolist())
            captions.extend(caption)

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

            # Encode the target image
            tar_img_feats.append(tar_img_feat.cpu())
            tar_txt_feats.append(tar_txt_feat.cpu())

        query_feats = torch.cat(query_feats, dim=0)
        tar_img_feats = torch.cat(tar_img_feats, dim=0)
        tar_txt_feats = torch.cat(tar_txt_feats, dim=0)

        query_feats = F.normalize(query_feats, dim=-1)
        tar_img_feats = F.normalize(tar_img_feats, dim=-1)
        tar_txt_feats = F.normalize(tar_txt_feats, dim=-1)

        ref_img_ids = [data_loader.dataset.pairid2ref[pair_id] for pair_id in pair_ids]
        tar_img_ids = [data_loader.dataset.pairid2tar[pair_id] for pair_id in pair_ids]

        ref_img_ids = torch.tensor(ref_img_ids, dtype=torch.long)
        tar_img_ids = torch.tensor(tar_img_ids, dtype=torch.long)

        if fabric.world_size > 1:
            # Gather tensors from every process
            query_feats = fabric.all_gather(query_feats)
            tar_img_feats = fabric.all_gather(tar_img_feats)
            tar_txt_feats = fabric.all_gather(tar_txt_feats)
            ref_img_ids = fabric.all_gather(ref_img_ids)
            tar_img_ids = fabric.all_gather(tar_img_ids)

            query_feats = einops.rearrange(query_feats, "d b q e -> (d b) q e")
            tar_img_feats = einops.rearrange(tar_img_feats, "d b q e -> (d b) q e")
            tar_txt_feats = einops.rearrange(tar_txt_feats, "d b e -> (d b) e")
            ref_img_ids = einops.rearrange(ref_img_ids, "d b -> (d b)")
            tar_img_ids = einops.rearrange(tar_img_ids, "d b -> (d b)")

        if fabric.global_rank == 0:
            tar_img_feats = tar_img_feats.mean(dim=1)
            query_feats = query_feats.mean(dim=1)
            sim_q2t_img = (query_feats @ tar_img_feats.t()).cpu().numpy()
            sim_q2t_txt = (query_feats @ tar_txt_feats.t()).cpu().numpy()

            sim_q2t = self.w_img * sim_q2t_img + self.w_txt * sim_q2t_txt

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
            json_dump(recalls, f"recalls_blip2-captions_{self.dataset}{self_sim}.json")

            print(
                f"Recalls saved in {Path.cwd()}/recalls_blip2-captions_{self.dataset}{self_sim}.json"
            )

        fabric.barrier()


class TestWebVidCoVRLateFusion:
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

            # Encode the reference image
            ref_img_feat = model.model.extract_features(
                {"image": ref_img}, mode="image"
            ).image_embeds_proj
            ref_img_feat = ref_img_feat.mean(dim=1)

            # Encode the caption
            assert isinstance(caption, list), "Caption must be a list of strings"
            text_input = [model.txt_processors["eval"](c) for c in caption]
            text_feat = model.model.extract_features(
                {"text_input": text_input}, mode="text"
            ).text_embeds_proj[:, 0, :]

            # Combine
            raw_combined_features = torch.cat((text_feat, ref_img_feat), -1)
            combined_features = F.relu(model.combiner_layer(raw_combined_features))
            dynamic_scalar = model.dynamic_scalar(raw_combined_features)
            query_feat = F.normalize(
                model.output_layer(combined_features)
                + dynamic_scalar * text_feat
                + (1 - dynamic_scalar) * ref_img_feat
            )
            query_feat = F.normalize(query_feat, dim=-1)
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
            tar_img_feats = einops.rearrange(tar_img_feats, "d b q e -> (d b) q e")
            ref_img_ids = einops.rearrange(ref_img_ids, "d b -> (d b)")
            tar_img_ids = einops.rearrange(tar_img_ids, "d b -> (d b)")

        if fabric.global_rank == 0:
            tar_img_feats = tar_img_feats.mean(dim=1)
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
            json_dump(recalls, f"recalls_late-fusion{self_sim}.json")

            print(f"Recalls saved in {Path.cwd()}/recalls_late-fusion{self_sim}.json")

        fabric.barrier()
