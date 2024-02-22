import argparse
import copy
import functools
import json
import logging
import os
import re
from collections.abc import Mapping, MutableMapping, Sequence
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path, PurePath
from typing import Any

import torch
import transformers
from packaging import version
from torch import nn
from torch.nn.modules.module import _IncompatibleKeys
from torch.serialization import MAP_LOCATION
from transformers import AutoTokenizer

from .clip_model import CLIP, CustomTextCLIP, convert_to_custom_text_state_dict, convert_weights_to_lp
from .coca_model import CoCa
from .loss import ClipLoss, CoCaLoss, DistillClipLoss, SigLipLoss
from .model import CLIPTextCfg, ImageTextModel, get_cast_dtype, resize_pos_embed, resize_text_pos_embed, \
    set_model_preprocess_cfg
from .openai import load_openai_model
from .precision import Precision
from .pretrained import download_pretrained, download_pretrained_from_hf, get_pretrained_cfg, \
    list_pretrained_tags_by_model
from .tokenizer import CLIP_TOKENIZER_TO_WRAP, DEFAULT_CONTEXT_LENGTH, HFTokenizer
from .transform import AugmentationCfg, PreprocessCfg, Transform, image_transform_v2, merge_preprocess_dict, \
    merge_preprocess_kwargs

HF_HUB_PREFIX = "hf-hub:"
_MODEL_CONFIG_PATHS = [Path(__file__).parent / f"model_configs/"]
_MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs


def _natural_key(string_: str) -> list[int]:  # Use `List` to show that it has the `__lt__` method.
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_.lower())]


def _rescan_model_configs() -> None:
    global _MODEL_CONFIGS

    config_ext = (".json",)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f"*{ext}"))

    for cf in config_files:
        with open(cf) as f:
            model_cfg = json.load(f)
        if all(a in model_cfg for a in ("embed_dim", "vision_cfg", "text_cfg")):
            _MODEL_CONFIGS[cf.stem] = model_cfg

    _MODEL_CONFIGS = {k: v for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))}


_rescan_model_configs()  # Initial population of the model config registry.


def list_models() -> Sequence[str]:
    """Enumerate the available model architectures based on the config files."""
    return list(_MODEL_CONFIGS.keys())


def add_model_config(path: str | PurePath) -> None:
    """ add model config path or file and update registry """
    if not isinstance(path, Path):
        path = Path(path)
    _MODEL_CONFIG_PATHS.append(path)
    _rescan_model_configs()


def get_model_config(model_name: str) -> MutableMapping[str, Any] | None:
    return deepcopy(_MODEL_CONFIGS[model_name]) if model_name in _MODEL_CONFIGS else None


def _get_hf_config(model_id: str, cache_dir: str | None = None) -> Mapping[str, Any]:
    config_path = download_pretrained_from_hf(model_id, filename="open_clip_config.json", cache_dir=cache_dir)
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


@functools.lru_cache(maxsize=4)
def get_tokenizer(model_name: str, context_length: int | None = None) -> HFTokenizer:
    if model_name.startswith(HF_HUB_PREFIX):
        model_name = model_name.removeprefix(HF_HUB_PREFIX)
        try:
            config = _get_hf_config(model_name)["model_cfg"]
        except FileNotFoundError:
            config = {"text_cfg": {"hf_tokenizer_name": model_name}}
    else:
        config = get_model_config(model_name)
        assert config is not None, f"No valid model config found for {model_name}."

    text_cfg = CLIPTextCfg(**config["text_cfg"])

    context_length = context_length or text_cfg.context_length

    wrapped_tokenizer_kwargs = {}
    if context_length:
        wrapped_tokenizer_kwargs["model_max_length"] = context_length

    if text_cfg.hf_tokenizer_name:
        wrapped_tokenizer = AutoTokenizer.from_pretrained(text_cfg.hf_tokenizer_name, **wrapped_tokenizer_kwargs)
    elif text_cfg.hf_model_name:
        wrapped_tokenizer = AutoTokenizer.from_pretrained(text_cfg.hf_model_name, **wrapped_tokenizer_kwargs)
    else:
        wrapped_tokenizer = CLIP_TOKENIZER_TO_WRAP
        if wrapped_tokenizer_kwargs:
            wrapped_tokenizer = copy.copy(wrapped_tokenizer)
            for k, v in wrapped_tokenizer_kwargs.items():
                setattr(wrapped_tokenizer, k, v)

    tokenizer_kwargs = text_cfg.tokenizer_kwargs or {}

    return HFTokenizer(wrapped_tokenizer, prefix=text_cfg.input_text_prefix, **tokenizer_kwargs)


def load_state_dict(checkpoint_path: str, map_location: MAP_LOCATION = "cpu") -> MutableMapping[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict", checkpoint)
    elif isinstance(checkpoint, torch.jit.ScriptModule):
        state_dict = checkpoint.state_dict()
        for key in ["input_resolution", "context_length", "vocab_size"]:
            state_dict.pop(key, None)
    else:
        state_dict = checkpoint
    if next(iter(state_dict.items()))[0].startswith("module."):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint(model: ImageTextModel, checkpoint_path: str, strict: bool = True) -> _IncompatibleKeys:
    if Path(checkpoint_path).suffix in {".npz", ".npy"}:
        from .big_vision import load_big_vision_weights
        load_big_vision_weights(model, checkpoint_path)  # noqa
        return _IncompatibleKeys([], [])

    state_dict = load_state_dict(checkpoint_path)

    state_dict = state_dict.get("model", state_dict)  # Unwrap the model state dict from the checkpoint state dict.

    if "logit_scale_hopfield" in state_dict:  # A CLOOB checkpoint.
        del state_dict["logit_scale_hopfield"]
        state_dict["logit_scale"] = state_dict.pop("logit_inv_tau")
        state_dict = {k.removeprefix("transformer.") if k.startswith("transformer.transformer.") else k: v
                      for k, v in state_dict.items()}

        for k in ["positional_embedding", "text_projection", "token_embedding.weight", "ln_final.weight",
                  "ln_final.bias"]:
            state_dict[k] = state_dict.pop(f"transformer.{k}")

    # Detect the old format and make it compatible with the new format.
    if "positional_embedding" in state_dict and not hasattr(model, "positional_embedding"):
        state_dict = convert_to_custom_text_state_dict(state_dict)

    # If we're loading a checkpoint with a longer context length.
    if "positional_embedding" in state_dict:
        state_dict["positional_embedding"] = \
            state_dict["positional_embedding"][:model.positional_embedding.shape[0]]
    elif "text.positional_embedding" in state_dict:
        state_dict["text.positional_embedding"] = \
            state_dict["text.positional_embedding"][:model.text.positional_embedding.shape[0]]
    elif "textual.posit_embed" in state_dict:
        state_dict["textual.posit_embed"] = state_dict["textual.posit_embed"][:model.textual.posit_embed.shape[0]]
        state_dict["textual.attn_mask"] = \
            state_dict["textual.attn_mask"][:model.textual.attn_mask.shape[0], :model.textual.attn_mask.shape[1]]

    if version.parse(transformers.__version__) >= version.parse("4.31"):
        # Certain text transformers no longer expect `position_ids` after transformers 4.31.
        position_id_key = "text.transformer.embeddings.position_ids"
        if position_id_key in state_dict and not hasattr(model, position_id_key):
            del state_dict[position_id_key]

    resize_pos_embed(state_dict, model)
    resize_text_pos_embed(state_dict, model)

    if getattr(model, "logit_bias", None) is not None and "logit_bias" not in state_dict:
        # This typically happens when we want to apply a SigLIP loss to a non-siglip-pretrained checkpoint.
        # A la SigLIP, we initialize it to cancel out the logit scale.
        #
        # Another option is to initialize it to zero, as you can view current score computation as having zero bias.
        # Though the loss computation changes, not sure if that assumption would be fine.
        state_dict["logit_bias"] = - state_dict["logit_scale"].exp()

    return model.load_state_dict(state_dict, strict=strict)


def create_model(
        model_name: str,
        pretrained: str | None = None,
        precision: Precision = "fp32",
        device: str | torch.device = "cpu",
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_text: bool = False,
        force_patch_dropout: float | None = None,
        force_image_size: int | tuple[int, int] | None = None,
        force_preprocess_cfg: Mapping[str, Any] | None = None,
        pretrained_image: bool = False,
        pretrained_hf_default: bool = True,
        cache_dir: str | None = None,
        require_pretrained: bool = False,
        tokenizer: HFTokenizer | None = None,
        **model_kwargs,
) -> ImageTextModel:
    force_preprocess_cfg = force_preprocess_cfg or {}
    preprocess_cfg = asdict(PreprocessCfg())

    if model_name.startswith(HF_HUB_PREFIX):
        model_id = model_name.removeprefix(HF_HUB_PREFIX)
        checkpoint_path = download_pretrained_from_hf(model_id, cache_dir=cache_dir)
        config = _get_hf_config(model_id, cache_dir)
        preprocess_cfg = merge_preprocess_dict(preprocess_cfg, config["preprocess_cfg"])
        model_cfg = config["model_cfg"]
        pretrained_hf = False  # Override. No need to load the original HF text weights.
    else:
        model_name = model_name.replace("/", "-")  # for callers using old naming with / in ViT names
        checkpoint_path = None
        model_cfg = None
        pretrained_hf = False

    if isinstance(device, str):
        device = torch.device(device)

    if pretrained and pretrained.lower() == "openai":
        logging.info(f"Loading pretrained {model_name} from OpenAI.")

        context_length = tokenizer.model_max_length if tokenizer else None

        model = load_openai_model(model_name, context_length=context_length, precision=precision, device=device,
                                  cache_dir=cache_dir, **model_kwargs)
        model.embed_dim = model.text_projection.shape[1]
    else:
        model_cfg = model_cfg or get_model_config(model_name)
        if model_cfg is None:
            logging.error(f"Model config for {model_name} not found; available models {list_models()}.")
            raise RuntimeError(f"Model config for {model_name} not found.")
        else:
            logging.info(f"Loaded {model_name} model config.")

        if force_quick_gelu:
            model_cfg["quick_gelu"] = True

        if force_patch_dropout is not None:
            model_cfg["vision_cfg"]["patch_dropout"] = force_patch_dropout

        if force_image_size is not None:
            model_cfg["vision_cfg"]["image_size"] = force_image_size

        is_timm_model = "timm_model_name" in model_cfg.get("vision_cfg", {})
        if pretrained_image:
            if is_timm_model:
                # pretrained weight loading for timm models set via vision_cfg
                model_cfg["vision_cfg"]["timm_model_pretrained"] = True
            else:
                raise ValueError("Pretrained image towers are currently only supported for timm models.")

        model_cfg.setdefault("text_cfg", {})
        text_cfg = model_cfg["text_cfg"]

        # cast_dtype set for fp16 and bf16 (manual mixed-precision), not set for "amp" or "pure" modes
        cast_dtype = get_cast_dtype(precision)
        is_hf_model = "hf_model_name" in text_cfg
        if is_hf_model:
            # load pretrained weights for HF text model IFF no CLIP weights being loaded
            model_cfg["text_cfg"]["hf_model_pretrained"] = pretrained_hf and not pretrained
        custom_text = model_cfg.pop("custom_text", False) or force_custom_text or is_hf_model

        if tokenizer:
            default_max_length = tokenizer.model_max_length - int(text_cfg.get("embed_cls", False))

            text_cfg.setdefault("context_length", default_max_length)
            text_cfg.setdefault("vocab_size", tokenizer.vocab_size)

            if "multimodal_cfg" in model_cfg:
                model_cfg["multimodal_cfg"].setdefault("context_length", default_max_length)
                model_cfg["multimodal_cfg"].setdefault("vocab_size", tokenizer.vocab_size)
        else:
            text_cfg.setdefault("context_length", DEFAULT_CONTEXT_LENGTH)

        model_cfg = dict(model_cfg, **model_kwargs)  # merge cfg dict w/ kwargs (kwargs overrides cfg)

        if custom_text:
            if is_hf_model:
                text_cfg.setdefault("hf_model_pretrained", pretrained_hf_default)

            if "multimodal_cfg" in model_cfg:
                model = CoCa(**model_cfg, cast_dtype=cast_dtype)
            else:
                model = CustomTextCLIP(**model_cfg, cast_dtype=cast_dtype)
        else:
            model = CLIP(**model_cfg, cast_dtype=cast_dtype)

        if getattr(model, "attn_mask", None) is None and (not hasattr(model, "text")
                                                          or getattr(model.text, "attn_mask", None) is None):
            # If there's no attention mask at all, it means that the model will attend to the padding tokens.
            # This is generally intended when the length of the input is fixed (e.g., when the text pooling strategy is
            # "last").
            # In consequence, we need to change the default padding strategy from "longest" to "max_length".
            # When the attention mask is set, it's typically causal, so it shouldn't attend to the padding tokens.
            # FIXME: potentially, there could be the case that a custom attention mask is set that depends on a fixed
            #  length padding strategy.
            #  Here we're assuming that this is not the case, and only change it when there's no attention mask.
            tokenizer.default_padding = "max_length"

        if precision in {"fp16", "bf16"}:
            dtype = torch.float16 if "fp16" in precision else torch.bfloat16
            # Manual mixed precision that matches the original OpenAI behavior.
            if is_timm_model:
                # FIXME: this is a bit janky, create timm based model in low-precision and
                #   then cast only LayerNormFp32 instances back to float32 so they don't break.
                #   Why? The convert_weights_to_lp fn only works with native models.
                model.to(device=device, dtype=dtype)
                from .transformer import LayerNormFp32

                def _convert_ln(m: nn.Module) -> None:
                    if isinstance(m, LayerNormFp32):
                        m.weight.data = m.weight.data.to(torch.float32)
                        m.bias.data = m.bias.data.to(torch.float32)

                model.apply(_convert_ln)
            else:
                model.to(device=device)
                convert_weights_to_lp(model, dtype=dtype)
        elif precision in {"pure_fp16", "pure_bf16"}:
            dtype = torch.float16 if "fp16" in precision else torch.bfloat16
            model.to(device=device, dtype=dtype)
        else:
            model.to(device=device)

        pretrained_loaded = False
        if pretrained:
            pretrained_cfg = get_pretrained_cfg(model_name, pretrained)
            if pretrained_cfg:
                checkpoint_path = download_pretrained(pretrained_cfg, cache_dir=cache_dir)
                preprocess_cfg = merge_preprocess_dict(preprocess_cfg, pretrained_cfg)
            elif os.path.exists(pretrained):
                checkpoint_path = pretrained
            else:
                checkpoint_path = None

            if checkpoint_path:
                logging.info(f"Loading pretrained {model_name} weights ({pretrained}).")
                load_checkpoint(model, checkpoint_path)
            else:
                error_str = (
                    f"Pretrained weights ({pretrained}) not found for model {model_name}."
                    f" Available pretrained tags ({list_pretrained_tags_by_model(model_name)}.")
                logging.warning(error_str)
                raise RuntimeError(error_str)
            pretrained_loaded = True
        elif checkpoint_path:
            logging.info(f"Loading pretrained {model_name} weights ({checkpoint_path}).")
            load_checkpoint(model, checkpoint_path)
            pretrained_loaded = True

        if require_pretrained and not pretrained_loaded:
            # callers of create_model_from_pretrained always expect pretrained weights
            raise RuntimeError(
                f"Pretrained weights were required for (model: {model_name}, pretrained: {pretrained}) but not loaded.")

        model.embed_dim = model_cfg["embed_dim"]

    if jit:
        model = torch.jit.script(model)

    # set image preprocessing configuration in model attributes for convenience
    if getattr(model.visual, "image_size", None):
        # use image_size set on model creation (via config or force_image_size arg)
        force_preprocess_cfg["size"] = model.visual.image_size
    set_model_preprocess_cfg(model, merge_preprocess_dict(preprocess_cfg, force_preprocess_cfg))

    return model


def create_loss(args: argparse.Namespace, model: ImageTextModel) -> nn.Module:
    if args.loss == "coca" or (args.loss == "info_nce" and "coca" in args.model.lower()):
        return CoCaLoss(
            model=model,
            caption_loss_weight=args.coca_caption_loss_weight,
            clip_loss_weight=args.coca_contrastive_loss_weight,
            do_gather=args.gather,
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            use_horovod=args.horovod,
        )
    elif args.distill:
        return DistillClipLoss(
            model=model,
            do_gather=args.gather,
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            use_horovod=args.horovod,
        )
    elif args.loss == "info_nce":
        extra_kwargs = ({} if args.multi_positive_strategy == "mean"
                        else {"multi_positive_strategy": args.multi_positive_strategy})
        return ClipLoss(
            model=model,
            do_gather=args.gather,
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            use_horovod=args.horovod,
            **extra_kwargs,
        )
    elif args.loss == "siglip":
        assert not args.horovod, "Horovod isn't currently supported for SigLip."
        return SigLipLoss(
            model=model,
            do_gather=args.gather,
            use_horovod=args.horovod,
        )
    else:
        raise ValueError(f"Unknown loss {args.loss}")


def create_model_and_transforms(
        model_name: str,
        pretrained: str | None = None,
        precision: Precision = "fp32",
        device: str | torch.device = "cpu",
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_text: bool = False,
        force_patch_dropout: float | None = None,
        force_image_size: int | tuple[int, int] | None = None,
        pretrained_image: bool = False,
        pretrained_hf_default: bool = True,
        image_mean: tuple[float, ...] | None = None,
        image_std: tuple[float, ...] | None = None,
        image_interpolation: str | None = None,
        image_resize_mode: str | None = None,  # Only effective for inference.
        aug_cfg: dict[str, Any] | AugmentationCfg | None = None,
        cache_dir: str | None = None,
        tokenizer: HFTokenizer | None = None,
        **model_kwargs,
) -> tuple[ImageTextModel, Transform, Transform]:
    force_preprocess_cfg = merge_preprocess_kwargs({}, mean=image_mean, std=image_std,
                                                   interpolation=image_interpolation, resize_mode=image_resize_mode)

    model = create_model(
        model_name,
        pretrained,
        precision=precision,
        device=device,
        jit=jit,
        force_quick_gelu=force_quick_gelu,
        force_custom_text=force_custom_text,
        force_patch_dropout=force_patch_dropout,
        force_image_size=force_image_size,
        force_preprocess_cfg=force_preprocess_cfg,
        pretrained_image=pretrained_image,
        pretrained_hf_default=pretrained_hf_default,
        cache_dir=cache_dir,
        tokenizer=tokenizer,
        **model_kwargs,
    )

    pp_cfg = PreprocessCfg(**model.visual.preprocess_cfg)

    train_transform = image_transform_v2(pp_cfg, is_train=True, aug_cfg=aug_cfg)
    val_transform = image_transform_v2(pp_cfg, is_train=False)

    return model, train_transform, val_transform


def create_model_from_pretrained(
        model_name: str,
        pretrained: str | None = None,
        precision: Precision = "fp32",
        device: str | torch.device = "cpu",
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_text: bool = False,
        force_image_size: int | tuple[int, int] | None = None,
        image_mean: tuple[float, ...] | None = None,
        image_std: tuple[float, ...] | None = None,
        image_interpolation: str | None = None,
        image_resize_mode: str | None = None,  # Only effective for inference.
        return_transform: bool = True,
        cache_dir: str | None = None,
        tokenizer: HFTokenizer | None = None,
        initial_temperature: float = 0.07,
        fit_temperature: bool = True,
        **model_kwargs,
) -> ImageTextModel | tuple[ImageTextModel, Transform]:
    force_preprocess_cfg = merge_preprocess_kwargs({}, mean=image_mean, std=image_std,
                                                   interpolation=image_interpolation, resize_mode=image_resize_mode)

    model = create_model(
        model_name,
        pretrained,
        precision=precision,
        device=device,
        jit=jit,
        force_quick_gelu=force_quick_gelu,
        force_custom_text=force_custom_text,
        force_image_size=force_image_size,
        force_preprocess_cfg=force_preprocess_cfg,
        cache_dir=cache_dir,
        tokenizer=tokenizer,
        require_pretrained=True,
        initial_temperature=initial_temperature,
        fit_temperature=fit_temperature,
        **model_kwargs,
    )

    if not return_transform:
        return model

    transform = image_transform_v2(PreprocessCfg(**model.visual.preprocess_cfg), is_train=False)

    return model, transform
