from typing import Any

import einops
import torch.nn.functional as F
from torch import nn
from transformers.models.bert.configuration_bert import BertConfig

from src.model.blip import create_vit, init_tokenizer, load_checkpoint
from src.model.med import BertModel
from src.tools.utils import print_dist


class BLIPCirVisualOnly(nn.Module):
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

        assert train_vit == True, "train_vit must be True for BLIPCirVisualOnly"
        self.train_vit = train_vit

        # Do not train text encoder
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        for p in self.text_proj.parameters():
            p.requires_grad = False

        self.temp = 0.07

    def forward(self, batch, fabric):
        ref_img, tar_feat, _, _ = batch

        device = ref_img.device

        # Encode the target image
        tar_feat = tar_feat.to(device)
        tar_img_feat = F.normalize(tar_feat, dim=-1)

        # Encode the reference image
        ref_img_embs = self.visual_encoder(ref_img)
        query_feat = F.normalize(self.vision_proj(ref_img_embs[:, 0, :]), dim=-1)

        if fabric.world_size > 1:
            # d: devices, b: batch size, e: embedding dim
            query_feat = fabric.all_gather(query_feat, sync_grads=True)
            query_feat = einops.rearrange(query_feat, "d b e -> (d b) e")

            tar_img_feat = fabric.all_gather(tar_img_feat, sync_grads=True)
            tar_img_feat = einops.rearrange(tar_img_feat, "d b e -> (d b) e")

        return self.loss(query_feat, tar_img_feat, self.temp)


def blip_cir_visual(model, ckpt_path, **kwargs):
    if ckpt_path:
        model, msg = load_checkpoint(model, ckpt_path)
        print_dist("missing keys:")
        print_dist(msg.missing_keys)
    return model
