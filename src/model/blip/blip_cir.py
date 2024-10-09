from typing import Any

import einops
import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.bert.configuration_bert import BertConfig

from src.model.blip.blip import create_vit, init_tokenizer, load_checkpoint
from src.model.blip.med import BertModel
from src.tools.utils import print_dist


class BLIPCir(nn.Module):
    def __init__(
        self,
        loss: Any,
        med_config="configs/med_config.json",
        image_size=384,
        vit="large",
        vit_grad_ckpt=True,
        vit_ckpt_layer=12,
        embed_dim=256,
        train_vit=False,
        si_ti_weight=1,
        si_tc_weight=0,
    ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()

        self.loss = loss

        self.visual_encoder, vision_width = create_vit(
            vit, image_size, vit_grad_ckpt, vit_ckpt_layer
        )
        self.tokenizer = init_tokenizer()
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)

        text_width = self.text_encoder.config.hidden_size

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.train_vit = train_vit
        if not self.train_vit:
            # Do not train visual encoder
            for p in self.visual_encoder.parameters():
                p.requires_grad = False

        for p in self.vision_proj.parameters():
            p.requires_grad = False

        self.temp = 0.07

        assert si_ti_weight + si_tc_weight > 0, "No loss term is enabled"
        self.si_ti_weight = si_ti_weight
        self.si_tc_weight = si_tc_weight

    def forward(self, batch, fabric):
        ref_img = batch["ref_img"]
        caption = batch["edit"]
        tar_img_feat = batch["tar_img_feat"]

        device = ref_img.device

        if self.train_vit:
            ref_img_embs = self.visual_encoder(ref_img)
        else:
            with torch.no_grad():
                ref_img_embs = self.visual_encoder(ref_img)

        # Encode the target image
        tar_img_feat = tar_img_feat.to(device)
        tar_img_feat = F.normalize(tar_img_feat, dim=-1)

        # Encode the reference image
        ref_img_atts = torch.ones(ref_img_embs.size()[:-1], dtype=torch.long).to(device)

        text = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(device)

        # Shift encoder
        encoder_input_ids = text.input_ids.clone()
        encoder_input_ids[:, 0] = self.tokenizer.enc_token_id
        query_si_embs = self.text_encoder(
            encoder_input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=ref_img_embs,
            encoder_attention_mask=ref_img_atts,
            return_dict=True,
        )
        query_si_feat = query_si_embs.last_hidden_state[:, 0, :]
        query_si_feat = F.normalize(self.text_proj(query_si_feat), dim=-1)

        if fabric.world_size > 1:
            # d: devices, b: batch size, e: embedding dim
            query_si_feat = fabric.all_gather(query_si_feat, sync_grads=True)
            query_si_feat = einops.rearrange(query_si_feat, "d b e -> (d b) e")

            tar_img_feat = fabric.all_gather(tar_img_feat, sync_grads=True)
            tar_img_feat = einops.rearrange(tar_img_feat, "d b e -> (d b) e")

        # s=source, t=target, i=image, c=caption, w=weight
        loss = 0
        if self.si_ti_weight > 0:
            si_ti_loss = self.loss(query_si_feat, tar_img_feat, self.temp)
            loss += si_ti_loss * self.si_ti_weight

        # Caption retrieval loss, only for WebVid-CoVR and CC-CoIR
        if self.si_tc_weight > 0:
            assert "tar_txt_feat" in batch, "tar_txt_feat is not in batch"
            tar_txt_feat = batch["tar_txt_feat"]

            if fabric.world_size > 1:
                tar_txt_feat = fabric.all_gather(tar_txt_feat, sync_grads=False)
                tar_txt_feat = einops.rearrange(tar_txt_feat, "d b e -> (d b) e")

            si_tc_loss = self.loss(query_si_feat, tar_txt_feat, self.temp)
            loss += si_tc_loss * self.si_tc_weight

        return loss


def blip_cir(model, ckpt_path, **kwargs):
    if ckpt_path:
        model, msg = load_checkpoint(model, ckpt_path)
        print_dist("missing keys:")
        print_dist(msg.missing_keys)
    return model
