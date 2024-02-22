#!/usr/bin/env python
import argparse
from collections.abc import Mapping

import torch

from open_clip.openai import load_openai_model
from training.argparse_with_defaults import ArgumentParserWithDefaults
from training.file_utils import pt_load
from training.utils import get_state_dict, compute_wise_state_dict


def parse_args() -> argparse.Namespace:
    parser = ArgumentParserWithDefaults(
        "Applies weight-space ensembles for fine-tuning (WiSE-FT or PAINT) on 2 CLIP checkpoints.",
        description="See https://arxiv.org/abs/2109.01903 and https://arxiv.org/abs/2208.05592 for more info.")
    parser.add_argument("input_path_or_url_or_name1", metavar="INPUT_FILE_OR_URL_1")
    parser.add_argument("input_path_or_url_or_name2", metavar="INPUT_FILE_OR_URL_2")
    parser.add_argument("output_path", metavar="OUTPUT_FILE")
    parser.add_argument("--weight-for-2", type=float, default=0.5)
    return parser.parse_args()


def _load_state_dict(path_or_url_or_name: str) -> Mapping[str, torch.Tensor]:
    if path_or_url_or_name.startswith("openai/"):
        return load_openai_model(path_or_url_or_name.removeprefix("openai/"), device="cpu",
                                 context_length=64).state_dict()
    else:
        return get_state_dict(pt_load(path_or_url_or_name))


def main() -> None:
    args = parse_args()
    state_dict1 = _load_state_dict(args.input_path_or_url_or_name1)
    state_dict2 = _load_state_dict(args.input_path_or_url_or_name2)
    torch.save(compute_wise_state_dict(state_dict1, state_dict2, weight_for_2=args.weight_for_2), args.output_path)


if __name__ == "__main__":
    main()
