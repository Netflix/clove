"""timm model adapter

Wraps timm (https://github.com/rwightman/pytorch-image-models) models for use as a vision tower in CLIP model.
"""
import copy
import logging
from collections import OrderedDict

import torch
from timm.models import named_apply
from timm.models.vision_transformer import get_init_weights_vit
from torch import nn

try:
    import timm
    from timm.models.layers import Mlp, to_2tuple

    try:
        # old timm imports < 0.8.1
        from timm.models.layers.attention_pool2d import RotAttentionPool2d
        from timm.models.layers.attention_pool2d import AttentionPool2d as AbsAttentionPool2d
    except ImportError:
        # new timm imports >= 0.8.1
        from timm.layers import RotAttentionPool2d
        from timm.layers import AttentionPool2d as AbsAttentionPool2d
except ImportError:
    timm = None

from .utils import freeze_batch_norm_2d


class VisionTransformerLikePoolingLayer(nn.Module):
    def __init__(self, global_pool: str, num_prefix_tokens: int):
        super().__init__()
        self.global_pool = global_pool
        self.num_prefix_tokens = num_prefix_tokens

    def forward(self, x, *args):
        if self.global_pool:
            return x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == "avg" else x[:, 0]
        else:
            return x


class TimmModel(nn.Module):
    """ timm model adapter
    """

    def __init__(
            self,
            model_name,
            embed_dim,
            image_size=224,
            pool="avg",
            proj="linear",
            proj_bias=False,
            drop=0.,
            drop_path=None,
            patch_drop=None,
            pretrained=False,
    ):
        super().__init__()
        if timm is None:
            raise RuntimeError("Please `pip install timm` to use timm models.")
        self.image_size = to_2tuple(image_size)

        # setup kwargs that may not be common across all models
        timm_kwargs = {}
        if drop_path is not None:
            timm_kwargs["drop_path_rate"] = drop_path
        if patch_drop is not None:
            timm_kwargs["patch_drop_rate"] = patch_drop

        CONV_2D_POOLING_OPTIONS = {"abs_attn", "rot_attn"}
        custom_pool = pool in (CONV_2D_POOLING_OPTIONS | {"attn"})
        if proj:
            assert proj in ("linear", "mlp", "none")
        extra_proj = proj in ("linear", "mlp")
        if not extra_proj and not custom_pool:
            # use network classifier head as projection if no proj specified and no custom pooling used
            # if projection is explicitly set to "none" will be pass through from network trunk
            proj_dim = 0 if proj == "none" else embed_dim
            self.trunk = timm.create_model(
                model_name,
                num_classes=proj_dim,
                global_pool=pool,
                pretrained=pretrained,
                **timm_kwargs,
            )
            prev_chs = embed_dim
            prev_global_pool = self.trunk.global_pool
            feat_size = None
        else:
            self.trunk = timm.create_model(
                model_name,
                pretrained=pretrained,
                **timm_kwargs,
            )
            feat_size = self.trunk.default_cfg.get("pool_size", None)
            feature_ndim = 2 if feat_size else 1
            if custom_pool:
                assert feature_ndim == (2 if pool in CONV_2D_POOLING_OPTIONS else 1)
                # if attn pooling used, remove both classifier and default pool
                reset_kwargs = dict(global_pool="")
            else:
                # reset global pool if pool config set, otherwise leave as network default
                reset_kwargs = dict(global_pool=pool) if pool else {}
            prev_global_pool = self.trunk.global_pool
            self.trunk.reset_classifier(0, **reset_kwargs)
            prev_chs = self.trunk.num_features

        head_layers = OrderedDict()

        # Add custom pooling to head
        if pool == "abs_attn":
            head_layers["pool"] = AbsAttentionPool2d(prev_chs, feat_size=feat_size, out_features=embed_dim)
            prev_chs = embed_dim
        elif pool == "rot_attn":
            head_layers["pool"] = RotAttentionPool2d(prev_chs, out_features=embed_dim)
            prev_chs = embed_dim
        elif pool == "attn":  # Only supported for ViT.
            block = copy.deepcopy(self.trunk.blocks[-1])
            named_apply(get_init_weights_vit(mode=""), block)

            head_layers["pool"] = nn.Sequential(OrderedDict([
                ("block", block),
                ("pool", VisionTransformerLikePoolingLayer(global_pool=prev_global_pool,
                                                           num_prefix_tokens=self.trunk.num_prefix_tokens))
            ]))
            # The feature size is preserved, so a projection should be used afterward.

        # NOTE attention pool ends with a projection layer, so proj should usually be set to "" if such pooling is used
        if proj == "linear":
            head_layers["drop"] = nn.Dropout(drop)
            head_layers["proj"] = nn.Linear(prev_chs, embed_dim, bias=proj_bias)
        elif proj == "mlp":
            head_layers["mlp"] = Mlp(prev_chs, 2 * embed_dim, embed_dim, drop=(drop, 0), bias=(True, proj_bias))
        else:
            assert not proj or proj == "none", f"Unknown projection type {proj}."

        self.head = nn.Sequential(head_layers)

    def lock(self, unlocked_groups: int = 0, freeze_bn_stats: bool = False) -> None:
        if unlocked_groups:
            # NOTE: a partial freeze requires the latest timm (main) branch and is subject to change.
            try:
                # FIXME import here until API stable and in an official release
                from timm.models.helpers import group_parameters, group_modules
            except ImportError:
                raise RuntimeError(
                    "Please install latest timm `pip install git+https://github.com/rwightman/pytorch-image-models`")
            matcher = self.trunk.group_matcher()
            gparams = group_parameters(self.trunk, matcher)
            max_layer_id = max(gparams.keys())
            max_layer_id = max_layer_id - unlocked_groups
            for group_idx in range(max_layer_id + 1):
                group = gparams[group_idx]
                for param in group:
                    self.trunk.get_parameter(param).requires_grad = False
            if freeze_bn_stats:
                g_modules = group_modules(self.trunk, matcher, reverse=True)
                g_modules = {k for k, v in g_modules.items() if v <= max_layer_id}
                freeze_batch_norm_2d(self.trunk, g_modules)
        else:
            # lock full model
            for param in self.trunk.parameters():
                param.requires_grad = False
            if freeze_bn_stats:
                freeze_batch_norm_2d(self.trunk)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        try:
            self.trunk.set_grad_checkpointing(enable)
        except Exception:
            logging.warning("grad checkpointing is not supported for this timm image tower, continuing without it…")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.trunk(x)
        x = self.head(x)
        return x
