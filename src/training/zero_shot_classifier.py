import argparse
from collections.abc import Callable, Collection
from string import Formatter

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from open_clip import get_autocast
from open_clip.model import ImageTextModel
from open_clip.tokenizer import HFTokenizer
from training.collate import MappingDataCollatorWithTokenization


Template = Callable[[str], str] | str

FORMATTER = Formatter()


def _instantiate_template(template: Template, class_name: str) -> str:
    if isinstance(template, str):
        # We do this because the class field within the template could be "{}", "{c}", or something else.
        if field_name := next(iter(FORMATTER.parse(template)))[1]:
            return template.format(**{field_name: class_name})
        else:
            return template.format(class_name)
    else:
        return template(class_name)


def build_zero_shot_classifier(model: ImageTextModel, tokenizer: HFTokenizer, class_names: Collection[str],
                               args: argparse.Namespace, templates: Collection[Template] = ("{}",),
                               dataset_name: str = "dataset") -> torch.Tensor:
    """Build zero-shot classifier weights by iterating over class names in batches."""

    if not class_names:
        return torch.empty(0).to(args.device)

    if not templates:
        raise ValueError("At least one template is required")

    texts = [{"text": _instantiate_template(template, c)} for c in class_names for template in templates]
    data_loader = DataLoader(texts, batch_size=args.eval_batch_size, num_workers=args.workers, pin_memory=True,  # noqa
                             persistent_workers=args.workers > 0,
                             collate_fn=MappingDataCollatorWithTokenization(tokenizer, keys_to_tokenize="text"))

    autocast = get_autocast(args.precision)

    with torch.inference_mode(), tqdm(unit=" class templates", total=len(texts),
                                      desc=f"Computing the {dataset_name} class representations") as p_bar:
        encoded_list = []

        for batch in data_loader:
            with autocast():
                encoded_list.append(model.encode_text(batch["text"].to(args.device, non_blocking=True)))
            p_bar.update(len(batch["text"]))

        return F.normalize(torch.cat(encoded_list).reshape(len(class_names), len(templates), -1).mean(dim=1))
