from collections.abc import Iterable, Mapping

import torch

from open_clip import ImageTextModel
from open_clip.utils import unwrap_model


def compute_retrieval_metrics(logits_per_name: Mapping[str, torch.Tensor], target_per_name: Mapping[str, torch.Tensor],
                              top_k: Iterable[int] = (1, 5, 10)) -> Mapping[str, float]:
    # FIXME: this does not scale past small eval datasets.
    #   `all_image_features @ all_text_features` will blow up memory and compute very quickly.
    target_per_name = {k: v.view(-1, 1) for k, v in target_per_name.items()}

    metrics = {}

    for name, logits in logits_per_name.items():
        ranks = logits.argsort(descending=True)
        preds = (ranks == target_per_name[name]).nonzero(as_tuple=True)[1]

        metrics[f"{name}_mean_rank"] = preds.mean(dtype=float).item() + 1
        # Without `+ 1`, so 0 is the min value:
        metrics[f"{name}_mean_rank_pct"] = preds.mean(dtype=float).item() / logits.shape[-1]
        metrics[f"{name}_median_rank"] = preds.median().item() + 1
        metrics[f"{name}_mrr"] = (1 / (preds + 1)).mean().item()

        for k in top_k:
            metrics[f"{name}_R@{k}"] = (preds < k).mean(dtype=float).item()

    return metrics


def compute_image_text_retrieval_metrics(model: ImageTextModel, image_features: torch.Tensor,
                                         text_features: torch.Tensor) -> Mapping[str, float]:
    return compute_retrieval_metrics(
        logits_per_name={
            "image_to_text": unwrap_model(model).compute_similarity(image_features, text_features),
            "text_to_image": unwrap_model(model).compute_similarity(text_features, image_features),
        },
        target_per_name={
            "image_to_text": torch.arange(len(image_features), device=text_features.device),
            "text_to_image": torch.arange(len(text_features), device=image_features.device),
        },
    )
