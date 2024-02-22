#!/usr/bin/env python
from collections.abc import Iterable

import pandas as pd
from tqdm.auto import tqdm

from training.__main__ import main

DEFAULT_BENCHMARKS = ("imagenet-val", "cb/wds/cars", "cb/wds/vtab/cifar10", "cb/wds/vtab/cifar100", "cb/wds/mnist",
                      "cb/wds/vtab/eurosat", "cb/wds/vtab/flowers", "cb/wds/vtab/dtd", "ucf101", "hmdb51")


def _get_args(model: str, pretrained: str, batch_size: int = 256, eval_benchmarks: Iterable[str] = DEFAULT_BENCHMARKS,
              resume: str | None = None, wise_ft: str | None = None,
              wise_ft_weight_for_2: float = 0.5) -> Iterable[str]:
    cmd = (f"--batch-size {batch_size} --model {model} --pretrained {pretrained}"
           f" --eval-benchmarks {' '.join(eval_benchmarks)}")

    if resume:
        cmd += f" --resume {resume}"

    if wise_ft:
        cmd += f" --wise-ft {wise_ft} --wise-ft-weight-for-2 {wise_ft_weight_for_2}"

    return cmd.split()


def _main() -> None:
    FINE_TUNED_URL = "https://github.com/Netflix/clove/releases/download/pretrained/clove_without_patching.pt"

    df = pd.DataFrame(
        {"method": name, **main(_get_args(**_args))}
        for name, _args in tqdm([
            ("pre-trained", {"model": "ViT-B-32", "pretrained": "openai"}),
            ("NegCLIP", {"model": "ViT-B-32", "pretrained": "negclip"}),
            ("REPLACE", {"model": "ViT-B-32", "pretrained": "sugarcrepe_replace_finetuned_e5"}),
            ("CLIP+\\textsc{CLoVe} w/o patching", {"model": "ViT-B-32", "pretrained": "openai",
                                                   "resume": FINE_TUNED_URL}),
            ("CLIP+\\textsc{CLoVe} (\\(\\alpha = .6\\))", {"model": "ViT-B-32", "pretrained": "openai",
                                                           "wise_ft": FINE_TUNED_URL,
                                                           "wise_ft_weight_for_2": 0.6}),
        ], desc="Evaluating methods")
    )

    df.set_index("method", inplace=True)
    df.index.name = ""

    df = df[[
        k
        for k in df.columns
        if ((k != "val/cb/wds/vtab/flowers" and k.endswith("-top1"))
            or (k == "val/cb/wds/vtab/flowers" and k.endswith("-mean_per_class_recall")))
    ]]
    df.rename(columns={
        "val/imagenet-val-top1": "ImageNet",
        "val/cb/wds/cars-top1": "Cars",
        "val/cb/wds/vtab/cifar10-top1": "CIFAR-10",
        "val/cb/wds/vtab/cifar100-top1": "CIFAR-100",
        "val/cb/wds/mnist-top1": "MNIST",
        "val/cb/wds/vtab/eurosat-top1": "EuroSAT",
        "val/cb/wds/vtab/flowers-mean_per_class_recall": "Flowers",
        "val/cb/wds/vtab/dtd-top1": "DTD",
        "val/ucf101-top1": "UCF101",
        "val/hmdb51-top1": "HMDB51",
    }, inplace=True)

    df["average"] = df.mean(axis=1)

    df *= 100

    style = (df.style
             .map_index(lambda v: "\\rot:--rwrap;", axis=1)  # FIXME: it's being escaped.
             # .hide(axis=0)
             .use({"hide_index_names": True})
             .format("{:00.1f}")
             .highlight_max(props="textbf:--rwrap;"))

    print(style.to_latex(label="tab:common-benchmark-results", caption="Zero-shot classification results.",
                         position_float="centering"))


if __name__ == "__main__":
    _main()
