import logging
from collections.abc import Mapping, MutableMapping
from typing import Any

import math
import torch
import torch.nn.functional as F
from torch import nn

from .model import CLIPTextCfg, CLIPVisionCfg, ImageTextModel, build_text_tower, build_vision_tower
from .transformer import Attention, TextTransformer, VisionTransformer, text_global_pool


class CLIP(ImageTextModel):
    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            cast_dtype: torch.dtype | None = None,
            initial_temperature: float = 0.07,
            init_logit_bias: float | None = None,
            fit_temperature: bool = True,
    ) -> None:
        super().__init__(embed_dim=embed_dim, vision_cfg=vision_cfg, text_cfg=text_cfg)

        self.visual = build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)

        text = build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.transformer = text.transformer
        self.context_length = text.context_length
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.text_pool_type = text.pool_type
        self.register_buffer("attn_mask", text.attn_mask, persistent=False)

        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / initial_temperature)), requires_grad=fit_temperature)
        if init_logit_bias is None:
            self.logit_bias = None
        else:
            self.logit_bias = nn.Parameter(torch.tensor(init_logit_bias, dtype=self.logit_scale.dtype))

    def lock_image_tower(self, unlocked_groups: int = 0, freeze_bn_stats: bool = False) -> None:
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable

    def encode_image(self, image: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        seq_len = text.shape[1]
        x = x + self.positional_embedding[:seq_len].to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=None if self.attn_mask is None else self.attn_mask[:seq_len, :seq_len])
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        x, _ = text_global_pool(x, text, self.text_pool_type)
        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                x = self.text_projection(x)
            else:
                x = x @ self.text_projection

        return F.normalize(x, dim=-1) if normalize else x

    def forward(self, image: torch.Tensor | None = None,
                text: torch.Tensor | None = None) -> Mapping[str, torch.Tensor]:
        output = {
            "image_features": None if image is None else self.encode_image(image),
            "text_features": None if text is None else self.encode_text(text),
            "logit_scale": self.logit_scale.exp(),
        }

        if self.logit_bias is not None:
            output["logit_bias"] = self.logit_bias

        return output

    def compute_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.logit_scale.exp() * x @ y.mT


class CustomTextCLIP(ImageTextModel):
    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            cast_dtype: torch.dtype | None = None,
            initial_temperature: float = 0.07,
            init_logit_bias: float | None = None,
            fit_temperature: bool = True,
    ) -> None:
        super().__init__(embed_dim=embed_dim, vision_cfg=vision_cfg, text_cfg=text_cfg)
        self.visual = build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        self.text = build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.context_length = self.text.context_length
        self.vocab_size = self.text.vocab_size

        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / initial_temperature)), requires_grad=fit_temperature)
        if init_logit_bias is None:
            self.logit_bias = None
        else:
            self.logit_bias = nn.Parameter(torch.tensor(init_logit_bias, dtype=self.logit_scale.dtype))

    def lock_image_tower(self, unlocked_groups: int = 0, freeze_bn_stats: bool = False) -> None:
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    def lock_text_tower(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True) -> None:
        self.text.lock(unlocked_layers, freeze_layer_norm)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)

    def encode_image(self, image: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        features = self.text(text)
        return F.normalize(features, dim=-1) if normalize else features

    def forward(self, image: torch.Tensor | None = None,
                text: torch.Tensor | None = None) -> Mapping[str, torch.Tensor]:
        output = {
            "image_features": None if image is None else self.encode_image(image),
            "text_features": None if text is None else self.encode_text(text),
            "logit_scale": self.logit_scale.exp(),
        }

        if self.logit_bias is not None:
            output["logit_bias"] = self.logit_bias

        return output

    def compute_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.logit_scale.exp() * x @ y.mT


def convert_weights_to_lp(module: nn.Module, dtype: torch.dtype = torch.float16) -> None:
    """Convert applicable module parameters to low precision (BF16 or FP16)."""

    def _convert_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            m.weight.data = m.weight.data.to(dtype)
            if m.bias is not None:
                m.bias.data = m.bias.data.to(dtype)

        if isinstance(m, (nn.MultiheadAttention, Attention)):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(m, attr)
                if tensor is not None:
                    tensor.data = tensor.data.to(dtype)

        if isinstance(m, (CLIP, TextTransformer)):
            # convert text nn.Parameter projections
            attr = getattr(m, "text_projection", None)
            if attr is not None:
                attr.data = attr.data.to(dtype)

        if isinstance(m, VisionTransformer):
            # convert vision nn.Parameter projections
            attr = getattr(m, "proj", None)
            if attr is not None:
                attr.data = attr.data.to(dtype)

    module.apply(_convert_weights)


convert_weights_to_fp16 = convert_weights_to_lp  # backwards compat


# used to maintain checkpoint compatibility
def convert_to_custom_text_state_dict(state_dict: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    if "text_projection" in state_dict:  # Old format state_dict. Move text tower -> .text
        return {
            ("text." if k.startswith(("text_projection", "positional_embedding", "token_embedding", "transformer",
                                      "ln_final")) else "") + k: v
            for k, v in state_dict.items()
        }
    return state_dict


def build_model_from_openai_state_dict(
        state_dict: MutableMapping[str, torch.Tensor],
        context_length: int | None = None,
        quick_gelu: bool = True,
        cast_dtype: torch.dtype = torch.float16,
        **model_kwargs,
) -> ImageTextModel:
    is_vit = "visual.proj" in state_dict

    if is_vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = sum(1 for k in state_dict if k.startswith("visual.") and k.endswith(".attn.in_proj_weight"))
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((len(state_dict["visual.positional_embedding"]) - 1) ** 0.5)
        image_size = vision_patch_size * grid_size
    else:
        vision_layers = tuple(len({k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}")})
                              for b in [1, 2, 3, 4])
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        total_vision_pos_embeds = len(state_dict["visual.attnpool.positional_embedding"])
        output_width = round((total_vision_pos_embeds - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == total_vision_pos_embeds
        image_size = output_width * 32

    state_dict_context_length = len(state_dict["positional_embedding"])

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = context_length or state_dict_context_length
    vocab_size = len(state_dict["token_embedding.weight"])
    transformer_width = len(state_dict["ln_final.weight"])
    transformer_heads = transformer_width // 64
    transformer_layers = len({k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")})

    vision_cfg = CLIPVisionCfg(
        layers=vision_layers,
        width=vision_width,
        patch_size=vision_patch_size,
        image_size=image_size,
    )
    text_cfg = CLIPTextCfg(
        context_length=context_length,
        vocab_size=vocab_size,
        width=transformer_width,
        heads=transformer_heads,
        layers=transformer_layers,
    )
    model = CLIP(
        embed_dim,
        vision_cfg=vision_cfg,
        text_cfg=text_cfg,
        quick_gelu=quick_gelu,  # OpenAI's models were trained with QuickGELU.
        cast_dtype=cast_dtype,
        **model_kwargs,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        state_dict.pop(key, None)

    convert_weights_to_fp16(model)  # OpenAI state dicts are partially converted to float16

    if context_length <= state_dict_context_length:
        state_dict["positional_embedding"] = state_dict["positional_embedding"][:context_length]
    else:
        randomly_initialized_embeddings = model.positional_embedding[state_dict_context_length:]
        logging.info(f"Adding extra {len(randomly_initialized_embeddings)} positional embeddings to the pre-trained"
                     f" ones.")
        state_dict["positional_embedding"] = torch.cat((state_dict["positional_embedding"],
                                                        randomly_initialized_embeddings))

    if getattr(model, "logit_bias", None) is not None and "logit_bias" not in state_dict:
        # This typically happens when we want to apply a SigLIP loss to a non-siglip-pretrained checkpoint.
        # A la SigLIP, we initialize it to cancel out the logit scale.
        state_dict["logit_bias"] = - state_dict["logit_scale"].exp()

    model.load_state_dict(state_dict)
    return model.eval()
