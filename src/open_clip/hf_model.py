"""HuggingFace model adapter

Wraps HuggingFace transformers (https://github.com/huggingface/transformers) models for use as a text tower in CLIP
model.
"""
import json
import logging
import os
import re
from typing import Literal

import torch
from huggingface_hub import snapshot_download
from torch import TensorType, nn

try:
    import transformers
    from transformers import AutoModel, AutoTokenizer, AutoConfig, PretrainedConfig
    from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, \
        BaseModelOutputWithPoolingAndCrossAttentions
except ImportError as e:
    transformers = None


    class BaseModelOutput:
        pass


    class PretrainedConfig:
        pass


# utils
def _camel2snake(s):
    return re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()


# TODO: ?last - for GPT-like models
_POOLERS = {}


def register_pooler(cls):
    """Decorator registering pooler class"""
    _POOLERS[_camel2snake(cls.__name__)] = cls
    return cls


@register_pooler
class MeanPooler(nn.Module):
    """Mean pooling"""

    @staticmethod
    def forward(x: BaseModelOutput, attention_mask: TensorType):
        masked_output = x.last_hidden_state * attention_mask.unsqueeze(-1)
        return masked_output.sum(dim=1) / attention_mask.sum(-1, keepdim=True)


@register_pooler
class MaxPooler(nn.Module):
    """Max pooling"""

    @staticmethod
    def forward(x: BaseModelOutput, attention_mask: TensorType):
        masked_output = x.last_hidden_state.masked_fill(attention_mask.unsqueeze(-1), -torch.inf)
        return masked_output.max(1).values


@register_pooler
class ClsPooler(nn.Module):
    """CLS token pooling"""

    def __init__(self, use_pooler_output=True):
        super().__init__()
        self.cls_token_position = 0
        self.use_pooler_output = use_pooler_output

    def forward(self, x: BaseModelOutput, attention_mask: TensorType):  # noqa
        if (self.use_pooler_output and
                isinstance(x, (BaseModelOutputWithPooling, BaseModelOutputWithPoolingAndCrossAttentions)) and
                x.pooler_output is not None
        ):
            return x.pooler_output

        return x.last_hidden_state[:, self.cls_token_position]


@register_pooler
class ClsLastHiddenStatePooler(nn.Module):
    """CLS token pooling
    NOTE: this is equivalent to ClsPooler above with use_pooler_output=False
    """

    def __init__(self):
        super().__init__()
        self.cls_token_position = 0

    def forward(self, x: BaseModelOutput, attention_mask: TensorType):  # noqa
        return x.last_hidden_state[:, self.cls_token_position]


HF_ARCH_DICT = {
    "bert": {
        "layer_attr": "layer",
        "token_embeddings_attr": "embeddings",
        "pooler": "cls_pooler",
    },
    "m2m_100": {
        "pooler": "cls_pooler",
    },
    "mt5": {
        "layer_attr": "block",
        "token_embeddings_attr": "embed_tokens",
        "pooler": "mean_pooler",
    },
    "roberta": {
        "layer_attr": "layer",
        "token_embeddings_attr": "embeddings",
        "pooler": "mean_pooler",
    },
    "t5": {
        "layer_attr": "block",
        "token_embeddings_attr": "embed_tokens",
        "pooler": "mean_pooler",
    },
    "xlm-roberta": {
        "layer_attr": "layer",
        "token_embeddings_attr": "embeddings",
        "pooler": "mean_pooler",
    },
}

ProjType = Literal["linear", "mlp"]
PoolerType = Literal["cls_last_hidden_state_pooler", "cls_pooler", "max_pooler", "mean_pooler"]


class HFTextEncoder(nn.Module):
    """HuggingFace model adapter"""
    output_tokens: torch.jit.Final[bool]

    def __init__(
            self,
            model_name_or_path: str,
            output_dim: int,
            config: PretrainedConfig | None = None,
            pooler_type: PoolerType | None = None,
            proj_type: ProjType | None = None,
            pretrained: bool = True,
            output_tokens: bool = False,
    ) -> None:
        super().__init__()
        self.output_tokens = output_tokens
        self.output_dim = output_dim

        # TODO: find better way to get this information
        uses_transformer_pooler = (pooler_type == "cls_pooler")

        if transformers is None:
            raise RuntimeError("Please run `pip install transformers` to use pre-trained HuggingFace models.")

        if config is None:
            self.config = AutoConfig.from_pretrained(model_name_or_path)
            create_func, model_args = (AutoModel.from_pretrained, model_name_or_path) if pretrained else (
                AutoModel.from_config, self.config)
            # TODO: do all model configs have this attribute? PretrainedConfig does so yes??
            if getattr(self.config, "is_encoder_decoder", False):
                self.transformer = create_func(model_args)
                self.transformer = self.transformer.encoder
            else:
                self.transformer = create_func(model_args, add_pooling_layer=uses_transformer_pooler)
        else:
            self.config = config
            self.transformer = AutoModel.from_config(config)

        if pooler_type is None:  # Try to infer the pooler type from the SentenceTransformers config.
            model_path = snapshot_download(model_name_or_path,
                                           ignore_patterns=["flax_model.msgpack", "rust_model.ot", "tf_model.h5"])
            sentence_transformers_modules_json_path = os.path.join(model_path, "modules.json")
            if os.path.exists(sentence_transformers_modules_json_path):
                with open(sentence_transformers_modules_json_path) as file:
                    modules_config = json.load(file)
                pooling_module_config = next((d
                                              for d in modules_config
                                              if d["type"] == "sentence_transformers.models.Pooling"),
                                             None)
                if pooling_module_config is not None:
                    pooling_config_path = os.path.join(model_path, pooling_module_config["path"], "config.json")
                    if os.path.exists(pooling_config_path):
                        with open(pooling_config_path) as file:
                            pooling_config = json.load(file)

                        assert len({k
                                    for k, v in pooling_config.items()
                                    if k != "word_embedding_dimension" and v}) == 1, \
                            ("This SentenceTransformer model applies multiple pooling types (or none?) but we support"
                             " exactly one.")

                        if pooling_config["pooling_mode_cls_token"]:
                            pooler_type = "cls_pooler"
                        elif pooling_config["pooling_mode_mean_tokens"]:
                            pooler_type = "mean_pooler"
                        elif pooling_config["pooling_mode_max_tokens"]:
                            pooler_type = "max_pooler"
                        elif pooling_config["pooling_mode_mean_sqrt_len_tokens"]:
                            raise NotImplementedError("Pooling mode 'mean_sqrt_len_tokens' is not supported.")
                        else:
                            raise ValueError("Unknown pooling mode from the SentenceTransformer config files.")
            else:
                # We only fall back to the arch config info if there's not a SentenceTransformer modules config because
                # otherwise we might get the wrong pooler type.
                pooler_type = HF_ARCH_DICT[self.config.model_type]["pooler"]

        # FIXME downstream users of OpenCLIP models use these attr, need to verify valid across all models
        self.vocab_size = getattr(self.config, "vocab_size", 0)
        self.context_length = getattr(self.config, "max_position_embeddings", 0)

        self.pooler = _POOLERS[pooler_type]()

        # `hidden_size` seems to be always defined in a way or another one (e.g., through a property or an attr map).
        d_model = self.config.hidden_size
        if (d_model == output_dim) and (proj_type is None):  # Do we always need a proj?
            self.proj = nn.Identity()
        elif proj_type == "linear":
            self.proj = nn.Linear(d_model, output_dim, bias=False)
        elif proj_type == "mlp":
            hidden_size = (d_model + output_dim) // 2
            self.proj = nn.Sequential(
                nn.Linear(d_model, hidden_size, bias=False),
                nn.GELU(),
                nn.Linear(hidden_size, output_dim, bias=False),
            )

    def forward(self, x: TensorType) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        attn_mask = (x != self.config.pad_token_id).long()
        out = self.transformer(input_ids=x, attention_mask=attn_mask)
        pooled_out = self.pooler(out, attn_mask)
        projected = self.proj(pooled_out)

        seq_len = out.last_hidden_state.shape[1]
        tokens = (
            out.last_hidden_state[:, torch.arange(seq_len) != self.pooler.cls_token_position]
            if type(self.pooler) == ClsPooler
            else out.last_hidden_state
        )

        if self.output_tokens:
            return projected, tokens
        return projected

    def lock(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        if not unlocked_layers:  # full freezing
            for n, p in self.transformer.named_parameters():
                p.requires_grad = "LayerNorm" in n.split(".") and not freeze_layer_norm
            return

        encoder = self.transformer.encoder if hasattr(self.transformer, "encoder") else self.transformer
        layer_list = getattr(encoder, HF_ARCH_DICT[self.config.model_type]["layer_attr"])
        logging.info(f"Unlocking {unlocked_layers}/{len(layer_list) + 1} layers from the HF model")
        embeddings = getattr(self.transformer, HF_ARCH_DICT[self.config.model_type]["token_embeddings_attr"])

        for module in [embeddings, *layer_list][:-unlocked_layers]:  # freeze layers
            for n, p in module.named_parameters():
                p.requires_grad = "LayerNorm" in n.split(".") and not freeze_layer_norm

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.gradient_checkpointing_enable()

    def init_parameters(self):
        pass
