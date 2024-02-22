"""CLIP Model

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
import copy
import functools
import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any

import math
import torch
import torch.nn.functional as F
from torch import nn

from .hf_model import HFTextEncoder, PoolerType, ProjType
from .modified_resnet import ModifiedResNet
from .precision import Precision
from .timm_model import TimmModel
from .transformer import LayerNorm, LayerNormFp32, QuickGELU, TextPoolType, TextTransformer, VisionTransformer
from .utils import to_2tuple


@dataclass
class CLIPVisionCfg:
    layers: tuple[int, int, int, int] | int = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: tuple[int, int] | int = 224

    ls_init_value: float | None = None  # layer scale initial value
    # What fraction of patches to dropout during training (0 would mean it's disabled and no patches dropped)
    # - 0.5 to 0.75 recommended in the paper for optimal results.
    patch_dropout: float = 0.
    # Whether to use dual patchnorm - would only apply the input layer norm on each patch,
    # as post-layer-norm already exists in the original CLIP ViT design.
    attentional_pool: bool = False  # whether to use attentional pooler in the last embedding layer
    attn_pooler_queries: int = 256  # n_queries for attentional pooler
    attn_pooler_heads: int = 8  # n heads for attentional_pooling
    no_ln_pre: bool = False  # disable pre transformer LayerNorm
    pos_embed_type: str = "learnable"
    final_ln_after_pool: bool = False  # apply final LayerNorm after pooling
    pool_type: str = "tok"
    output_tokens: bool = False
    act_kwargs: dict | None = None
    norm_kwargs: dict | None = None

    timm_model_name: str | None = None  # a valid model name overrides layers, width, patch_size
    timm_model_pretrained: bool = False  # Use (imagenet) pretrained weights for the named model.
    timm_pool: str = "avg"  # feature pooling for timm model ("abs_attn", "rot_attn", "avg", "")
    timm_proj: str | None = "linear"  # linear projection for timm model output ("linear", "mlp", "none")
    timm_proj_bias: bool = False  # enable bias final projection
    timm_drop: float = 0.  # head dropout
    timm_drop_path: float | None = None  # backbone stochastic depth


@dataclass
class CLIPTextCfg:
    context_length: int | None = None
    vocab_size: int = 49408
    hf_tokenizer_name: str | None = None
    tokenizer_kwargs: dict | None = None

    width: int = 512
    heads: int = 8
    layers: int = 12
    mlp_ratio: float = 4.0
    ls_init_value: float | None = None  # layer scale initial value
    embed_cls: bool = False
    pad_id: int = 0
    no_causal_mask: bool = False  # disable causal masking
    final_ln_after_pool: bool = False  # apply final LayerNorm after pooling
    pool_type: TextPoolType = "argmax"
    proj_bias: bool = False
    output_tokens: bool = False
    act_kwargs: dict = None
    norm_kwargs: dict = None

    input_text_prefix: str = ""

    # HuggingFace-specific text tower config
    hf_model_name: str | None = None
    hf_model_pretrained: bool = True
    hf_proj_type: ProjType | None = None
    hf_pooler_type: PoolerType | None = None


def get_cast_dtype(precision: Precision) -> torch.dtype | None:
    if precision == "bf16":
        return torch.bfloat16
    elif precision == "fp16":
        return torch.float16
    else:
        return None


def get_input_dtype(precision: Precision) -> torch.dtype | None:
    if precision in {"bf16", "pure_bf16"}:
        return torch.bfloat16
    elif precision in {"fp16", "pure_fp16"}:
        return torch.float16
    else:
        return None


def build_vision_tower(
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        quick_gelu: bool = False,
        cast_dtype: torch.dtype | None = None,
) -> nn.Module:
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    # OpenAI's models are pretrained w/ QuickGELU but native nn.GELU is both faster and more memory-efficient in recent
    # PyTorch releases (>= 1.10). NOTE: timm models always use native GELU regardless of quick_gelu flag.
    act_layer = QuickGELU if quick_gelu else nn.GELU

    if vision_cfg.timm_model_name:
        visual = TimmModel(
            vision_cfg.timm_model_name,
            pretrained=vision_cfg.timm_model_pretrained,
            pool=vision_cfg.timm_pool,
            proj=vision_cfg.timm_proj,
            proj_bias=vision_cfg.timm_proj_bias,
            drop=vision_cfg.timm_drop,
            drop_path=vision_cfg.timm_drop_path,
            patch_drop=vision_cfg.patch_dropout if vision_cfg.patch_dropout > 0 else None,
            embed_dim=embed_dim,
            image_size=vision_cfg.image_size,
        )
    elif isinstance(vision_cfg.layers, (tuple, list)):
        vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
        visual = ModifiedResNet(
            layers=vision_cfg.layers,
            output_dim=embed_dim,
            heads=vision_heads,
            image_size=vision_cfg.image_size,
            width=vision_cfg.width,
        )
    else:
        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        if vision_cfg.norm_kwargs:
            norm_layer = functools.partial(norm_layer, **vision_cfg.norm_kwargs)
        if vision_cfg.act_kwargs is not None:
            act_layer = functools.partial(act_layer, **vision_cfg.act_kwargs)

        visual = VisionTransformer(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            ls_init_value=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            attentional_pool=vision_cfg.attentional_pool,
            attn_pooler_queries=vision_cfg.attn_pooler_queries,
            attn_pooler_heads=vision_cfg.attn_pooler_heads,
            pos_embed_type=vision_cfg.pos_embed_type,
            no_ln_pre=vision_cfg.no_ln_pre,
            final_ln_after_pool=vision_cfg.final_ln_after_pool,
            pool_type=vision_cfg.pool_type,
            output_tokens=vision_cfg.output_tokens,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

    return visual


def build_text_tower(
        embed_dim: int,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        cast_dtype: torch.dtype | None = None,
) -> nn.Module:
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)

    if text_cfg.hf_model_name:
        text = HFTextEncoder(
            text_cfg.hf_model_name,
            output_dim=embed_dim,
            proj_type=text_cfg.hf_proj_type,
            pooler_type=text_cfg.hf_pooler_type,
            pretrained=text_cfg.hf_model_pretrained,
            output_tokens=text_cfg.output_tokens,
        )
    else:
        act_layer = QuickGELU if quick_gelu else nn.GELU
        norm_layer = LayerNormFp32 if cast_dtype in {torch.float16, torch.bfloat16} else LayerNorm
        if text_cfg.norm_kwargs:
            norm_layer = functools.partial(norm_layer, **text_cfg.norm_kwargs)
        if text_cfg.act_kwargs is not None:
            act_layer = functools.partial(act_layer, **text_cfg.act_kwargs)

        text = TextTransformer(
            context_length=text_cfg.context_length,
            vocab_size=text_cfg.vocab_size,
            width=text_cfg.width,
            heads=text_cfg.heads,
            layers=text_cfg.layers,
            mlp_ratio=text_cfg.mlp_ratio,
            ls_init_value=text_cfg.ls_init_value,
            output_dim=embed_dim,
            embed_cls=text_cfg.embed_cls,
            no_causal_mask=text_cfg.no_causal_mask,
            pad_id=text_cfg.pad_id,
            pool_type=text_cfg.pool_type,
            proj_bias=text_cfg.proj_bias,
            output_tokens=text_cfg.output_tokens,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
    return text


class ImageTextModel(nn.Module, ABC):
    visual: nn.Module
    logit_scale: torch.Tensor

    def __init__(self, embed_dim: int, vision_cfg: CLIPVisionCfg, text_cfg: CLIPTextCfg) -> None:  # noqa
        super().__init__()

    @abstractmethod
    def encode_image(self, image: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def encode_text(self, text: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, image: torch.Tensor | None = None,
                text: torch.Tensor | None = None) -> Mapping[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def compute_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes the similarity between each of the normalized features x and each of the normalized features y, for
        the last two dimensions (it behaves element-wise for the rest).

        The features have shape `(*, E)`, with at least two dimensions.
        """
        raise NotImplementedError

    def compute_similarity_pairwise(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes the similarity between the normalized features x and y.

        The similarity is computed element-wise, taking each pair of elements. Thus, the features should have the same
        shape, with at least one dimension.
        """
        return self.compute_similarity(x.unsqueeze(1), y.unsqueeze(1)).squeeze(1).squeeze(1)


def trace_model(model: ImageTextModel, batch_size: int = 256,
                device: torch.device = torch.device("cpu")) -> torch.ScriptModule:
    model.eval()
    image_size = model.visual.image_size
    example_images = torch.ones((batch_size, 3, image_size, image_size), device=device)
    example_text = torch.zeros((batch_size, model.context_length), dtype=torch.int, device=device)
    model = torch.jit.trace_module(
        model,
        inputs=dict(
            forward=(example_images, example_text),
            encode_text=(example_text,),
            encode_image=(example_images,)
        ))
    model.visual.image_size = image_size
    return model


def resize_pos_embed(state_dict: MutableMapping[str, Any], model: ImageTextModel, interpolation: str = "bicubic",
                     antialias: bool = True) -> None:
    # Rescale the grid of position embeddings when loading from a state dict.
    old_pos_embed = state_dict.get("visual.positional_embedding", None)
    if old_pos_embed is None or not hasattr(model.visual, "grid_size"):
        return
    grid_size = to_2tuple(model.visual.grid_size)
    extra_tokens = 1  # FIXME detect different token configs (ie no class token, or more)
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    if new_seq_len == old_pos_embed.shape[0]:
        return

    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed
    old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))

    logging.info("Resizing position embedding grid-size from %s to %s", old_grid_size, grid_size)
    pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1).permute(0, 3, 1, 2)
    pos_emb_img = F.interpolate(
        pos_emb_img,
        size=grid_size,
        mode=interpolation,
        antialias=antialias,
        align_corners=False,
    )
    pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)[0]
    state_dict["visual.positional_embedding"] = (pos_emb_img if pos_emb_tok is None
                                                 else torch.cat([pos_emb_tok, pos_emb_img]))


def resize_text_pos_embed(state_dict, model, interpolation: str = "linear", antialias: bool = False) -> None:
    old_pos_embed = state_dict.get("positional_embedding", None)
    if old_pos_embed is None:
        return
    # FIXME add support for text cls_token
    model_pos_embed = getattr(model, "positional_embedding", None)
    if model_pos_embed is None:
        model_pos_embed = getattr(model.text, "positional_embedding", None)

    old_num_pos = old_pos_embed.shape[0]
    old_width = old_pos_embed.shape[1]
    num_pos = model_pos_embed.shape[0]
    width = model_pos_embed.shape[1]
    assert old_width == width, "text pos_embed width changed!"
    if old_num_pos == num_pos:
        return

    logging.info(f"Resizing text position embedding num_pos from {old_num_pos} to {num_pos}")
    old_pos_embed = old_pos_embed.reshape(1, old_num_pos, old_width).permute(0, 2, 1)
    old_pos_embed = F.interpolate(
        old_pos_embed,
        size=num_pos,
        mode=interpolation,
        antialias=antialias,
        align_corners=False,
    )
    old_pos_embed = old_pos_embed.permute(0, 2, 1)[0]
    new_pos_embed = old_pos_embed

    state_dict["positional_embedding"] = new_pos_embed


def get_model_preprocess_cfg(model: ImageTextModel) -> Mapping[str, Any]:
    preprocess_cfg = getattr(model.visual, "preprocess_cfg", {})
    if not preprocess_cfg:
        # use separate legacy attributes if preprocess_cfg dict not found
        size = getattr(model.visual, "image_size")
        if size is not None:
            preprocess_cfg["size"] = size
        mean = getattr(model.visual, "image_mean", None)
        if mean is not None:
            preprocess_cfg["mean"] = mean
        std = getattr(model.visual, "image_std", None)
        if std is not None:
            preprocess_cfg["std"] = std
    return preprocess_cfg


def set_model_preprocess_cfg(model: ImageTextModel, preprocess_cfg: Mapping[str, Any]) -> None:
    model.visual.image_mean = preprocess_cfg["mean"]  # legacy attribute, keeping for bwd compat
    model.visual.image_std = preprocess_cfg["std"]  # legacy attribute, keeping for bwd compat
    model.visual.preprocess_cfg = copy.deepcopy(preprocess_cfg)  # new attr, package all pp cfg as dict


def get_model_tokenize_cfg(model: ImageTextModel) -> Mapping[str, Any]:
    text_module = getattr(model, "text", model)
    cfg = {}
    context_length = getattr(text_module, "context_length", None)
    if context_length is not None:
        cfg["context_length"] = context_length
    vocab_size = getattr(text_module, "vocab_size", None)
    if vocab_size is not None:
        cfg["vocab_size"] = vocab_size
    return cfg
