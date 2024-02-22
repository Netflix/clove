"""OpenAI pretrained model functions.

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""

import os
from collections.abc import Sequence

import torch

from .clip_model import build_model_from_openai_state_dict, convert_weights_to_lp
from .constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from .model import ImageTextModel, get_cast_dtype
from .precision import Precision
from .pretrained import download_pretrained_from_url, get_pretrained_url, list_pretrained_models_by_tag


def list_openai_models() -> Sequence[str]:
    """Returns the names of available CLIP models."""
    return list_pretrained_models_by_tag("openai")


def load_openai_model(
        name: str,
        context_length: int | None = None,
        precision: Precision | None = None,
        device: str | torch.device | None = None,
        cache_dir: str | None = None,
        **model_kwargs,
) -> ImageTextModel:
    """Load a CLIP model.

    Parameters
    ----------
    name: A model name listed by `clip.available_models()`,
        or the path to a model checkpoint containing the state_dict
    context_length: The max number of positional embeddings to use for the model.
    precision: Model precision, if None defaults to "fp32" if `device == "cpu"` else "fp16".
    device: The device to put the loaded model
    cache_dir: The directory to cache the downloaded model weights

    Returns
    -------
    model : torch.nn.Module
        The CLIP model
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if precision is None:
        precision = "fp32" if device == "cpu" else "fp16"

    pretrained_url = get_pretrained_url(name, "openai")
    if pretrained_url:
        model_path = download_pretrained_from_url(pretrained_url, cache_dir=cache_dir)
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {list_openai_models()}")

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        model = None
        state_dict = torch.load(model_path, map_location="cpu")

    # Build a non-JIT model from the OpenAI jitted model state dict.
    cast_dtype = get_cast_dtype(precision)
    try:
        model = build_model_from_openai_state_dict(state_dict or model.state_dict(), context_length=context_length,
                                                   cast_dtype=cast_dtype, **model_kwargs)
    except KeyError:
        sd = {k[7:]: v for k, v in state_dict["state_dict"].items()}
        model = build_model_from_openai_state_dict(sd, context_length=context_length, cast_dtype=cast_dtype,
                                                   **model_kwargs)

    model = model.to(device)

    # The model from OpenAI state dict is in manually-cast FP16 mode.
    # It must be converted for AMP/FP32/BF16 use.
    # FIXME: support pure FP16/BF16 precision modes.
    if precision != "fp16":
        model.float()
        if precision == "bf16":
            # For BF16, convert back to low precision.
            convert_weights_to_lp(model, dtype=torch.bfloat16)

    # add mean / std attributes for consistency with OpenCLIP models
    model.visual.image_mean = OPENAI_DATASET_MEAN
    model.visual.image_std = OPENAI_DATASET_STD
    return model
