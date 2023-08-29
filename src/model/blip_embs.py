import torch
from torch import nn

from src.model.blip import create_vit, init_tokenizer, load_checkpoint
from src.model.med import BertConfig, BertModel


class BLIPEmbs(nn.Module):
    def __init__(
        self,
        med_config="configs/med_config.json",
        image_size=384,
        vit="base",
        vit_grad_ckpt=False,
        vit_ckpt_layer=0,
        embed_dim=256,
        queue_size=57600,
        negative_all_rank=False,
    ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()

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

        self.queue_size = queue_size
        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.negative_all_rank = negative_all_rank


def blip_embs(pretrained="", **kwargs):
    model = BLIPEmbs(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
        print("missing keys:")
        print(msg.missing_keys)
        assert len(msg.missing_keys) == 0, "Missing keys!"
    return model
