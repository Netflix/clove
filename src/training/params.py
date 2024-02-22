import argparse
import ast
import multiprocessing
import os
from collections.abc import Iterable, Mapping
from typing import Any

import torch

from open_clip import PRECISION_OPTIONS
from training.argparse_with_defaults import ArgumentParserWithDefaults
from training.data import get_registered_datasets

REGISTERED_EVAL_BENCHMARK = sorted(get_registered_datasets())
DEFAULT_EVAL_BENCHMARKS = sorted(set(REGISTERED_EVAL_BENCHMARK) - {"imagenet-v2", "youcook2"})

_DATASET_SHORT_NAMES = {
    "cc3m": "[FILL_PATH]/cc3m/train_wds/{00000..00331}.tar",
    "cc12m": "[FILL_PATH]/cc12m/{00000..01242}.tar",
    "coco": "[FILL_PATH]/coco/{00000..00059}.tar",
    "coyo": "[FILL_PATH]/coyo-700m/data/{00000..74751}.tar",
    "laion": "[FILL_PATH]/laion400m/data/{00000..41407}.tar",
    "laion-coco": "[FILL_PATH]/laion-coco/data/{00000..66303}.tar",
}


def _get_default_params(args: argparse.Namespace) -> Mapping[str, Any]:
    # Some params come from the paper (https://arxiv.org/pdf/2103.00020.pdf).
    return {
        "beta2": 0.98 if "vit" in args.model.lower() else 0.999,
        "eps": 1e-6 if "vit" in args.model.lower() else 1e-8,
        "initial_temperature": .1 if args.loss == "siglip" else 0.07,
    }


class ParseKwargs(argparse.Action):
    def __call__(self, parser: argparse.ArgumentParser, namespace: argparse.Namespace, values: Iterable[str],
                 option_string: str | None = None) -> None:
        kw = {}
        for value in values:
            key, value = value.split("=")
            try:
                kw[key] = ast.literal_eval(value)
            except ValueError:
                kw[key] = str(value)  # fallback to string (avoid need to escape on command line)
        setattr(namespace, self.dest, kw)


def create_parser() -> argparse.ArgumentParser:
    parser = ArgumentParserWithDefaults()
    parser.add_argument("--train-data", help="Path to file(s) with training data. When using webdataset, multiple data"
                                             " sources can be combined using the `::` separator.")
    parser.add_argument("--train-data-upsampling-factors",
                        help="When using multiple data sources with webdataset and sampling with replacement, this can"
                             " be used to upsample specific data sources. Similar to --train-data, this should be a"
                             " string with as many numbers as there are data sources, separated by `::` (e.g."
                             " 1::2::0.5). By default, datapoints are sampled uniformly regardless of the dataset"
                             " sizes.")
    parser.add_argument("--val-data",
                        default="[PATH]/cc3m/val_wds/{00000..00001}.tar",
                        help="Path to file(s) with validation data")
    parser.add_argument("--train-num-samples", type=int,
                        help="Number of samples in dataset. Required for webdataset if not available in info file.")
    parser.add_argument("--val-num-samples", type=int,
                        help="Number of samples in dataset. Useful for webdataset if not available in info file.")
    parser.add_argument("--dataset-type", choices=["auto", "csv", "synthetic", "webdataset"], default="auto",
                        help="Which type of dataset to process.")
    parser.add_argument("--dataset-resampled", action="store_true",
                        help="Whether to use sampling with replacement for webdataset shard selection.")
    parser.add_argument("--shuffle-train-with-buffer", action="store_true",
                        help="Whether to shuffle the training shards with a buffer, as opposed to shuffling all the"
                             " shards at once (the default).")
    parser.add_argument("--csv-separator", default="\t", help="For csv-like datasets, which separator to use.")
    parser.add_argument("--csv-img-key", default="filepath",
                        help="For csv-like datasets, the name of the key for the image paths.")
    parser.add_argument("--csv-caption-key", default="title",
                        help="For csv-like datasets, the name of the key for the captions.")
    parser.add_argument("--imagenet-val",
                        default="[PATH]/imagenet/ILSVRC2012_img_val/",
                        help="Path to the ImageNet val set for conducting zero-shot evaluation.")
    parser.add_argument("--imagenet-v2", default="/mnt",
                        help="Path to the ImageNet v2 for conducting zero-shot evaluation.")
    parser.add_argument("--eval-benchmarks", nargs="*", default=DEFAULT_EVAL_BENCHMARKS,
                        help="Which benchmarks to use for zero-shot evaluation. It could be one from `clip_benchmark`"
                             f" (use the prefix 'cb/') or one of {REGISTERED_EVAL_BENCHMARK}.")
    parser.add_argument("--no-use-templates", dest="use_templates", action="store_false",
                        help="Whether to use templates for the classification evaluations.")
    parser.add_argument("--logs", default="logs",
                        help="Where to store tensorboard logs. Use None to avoid storing logs.")
    parser.add_argument("--log-local", action="store_true",
                        help="log files on local master, otherwise global master only.")
    parser.add_argument("--name",
                        help="Optional identifier for the experiment when storing logs. Otherwise use current time.")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count() // max(torch.cuda.device_count(), 1),
                        help="Number of data loader workers per GPU.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size per GPU.")
    parser.add_argument("--eval-batch-size", type=int, help="Batch size per GPU for evaluation. By default, it's the"
                                                            " same as the training batch size.")
    parser.add_argument("--epochs", type=int, default=32, help="Number of epochs to train for.")
    parser.add_argument("--epochs-cooldown", type=int,
                        help="When scheduler w/ cooldown used, perform cooldown from total_epochs - cooldown_epochs"
                             " onwards.")
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--lr", default=5e-4, type=float, help="Learning rate.")
    parser.add_argument("--beta1", default=0.9, type=float, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")
    parser.add_argument("--warmup", type=int, default=10000, help="Number of steps to warmup for.")
    parser.add_argument("--use-bn-sync", action="store_true", help="Whether to use batch norm sync.")
    parser.add_argument("--skip-scheduler", action="store_true", help="Use this flag to skip the learning rate decay.")
    parser.add_argument("--lr-scheduler", choices=["cosine", "const", "const-cooldown", "lr-finder"], default="cosine",
                        help="LR scheduler. One of: 'cosine', 'const' (constant), 'const-cooldown' (constant w/"
                             " cooldown)")
    parser.add_argument("--lr-cooldown-end", type=float, default=0.0, help="End learning rate for cooldown schedule")
    parser.add_argument("--lr-cooldown-power", type=float, default=1.0,
                        help="Power for polynomial cooldown schedule. Default: linear decay.")
    parser.add_argument("--save-frequency", type=int, default=1, help="How often to save checkpoints.")
    parser.add_argument("--save-most-recent", action="store_true",
                        help="Always save the most recent model trained to epoch_latest.pt.")
    parser.add_argument("--zeroshot-frequency", type=int, default=1, help="How often to run zero shot.")
    parser.add_argument("--val-frequency", type=int, default=1, help="How often to run evaluation with val data.")
    parser.add_argument("--resume", help="path to latest checkpoint")
    parser.add_argument("--precision", choices=PRECISION_OPTIONS, default="amp", help="Floating point precision.")
    parser.add_argument("--model", default="RN50", help="Name of the vision backbone to use.")
    parser.add_argument("--pretrained", help="Use a pretrained CLIP model weights with the specified tag or file path.")
    parser.add_argument("--pretrained-image", action="store_true",
                        help="Load pretrained weights for image tower backbone if available.")
    parser.add_argument("--lock-image", action="store_true", help="Lock full image tower by disabling gradients.")
    parser.add_argument("--lock-image-unlocked-groups", type=int, default=0,
                        help="Leave last n image tower layer groups unlocked.")
    parser.add_argument("--lock-image-freeze-bn-stats", action="store_true",
                        help="Freeze BatchNorm running stats in image tower for any locked layers.")
    parser.add_argument("--image-mean", type=float, nargs="+", metavar="MEAN",
                        help="Override default image mean value of dataset")
    parser.add_argument("--image-std", type=float, nargs="+", metavar="STD",
                        help="Override default image std deviation of of dataset")
    parser.add_argument("--image-interpolation", choices=["bicubic", "bilinear", "random"],
                        help="Override default image resize interpolation")
    parser.add_argument("--image-resize-mode", choices=["shortest", "longest", "squash"],
                        help="Override default image resize (& crop) mode during inference")
    parser.add_argument("--aug-cfg", nargs="*", default={}, action=ParseKwargs)
    parser.add_argument("--grad-checkpointing", action="store_true", help="Enable gradient checkpointing.")
    parser.add_argument("--no-gather", dest="gather", action="store_false",
                        help="Gather all model outputs before computing the loss.")
    parser.add_argument("--no-local-loss", dest="local_loss", action="store_true",
                        help="calculate loss w/ local features @ global (instead of realizing full global @ global"
                             " matrix)")
    parser.add_argument("--no-gather-with-grad", dest="gather_with_grad", action="store_false",
                        help="Disable full distributed gradient for feature gather.")
    parser.add_argument("--force-image-size", type=int, nargs="+", help="Override default image size")
    parser.add_argument("--force-quick-gelu", action="store_true",
                        help="Force use of QuickGELU activation for non-OpenAI transformer models.")
    parser.add_argument("--force-patch-dropout", type=float,
                        help="Override the patch dropout during training, for fine tuning with no dropout near the end"
                             " as in the paper", )
    parser.add_argument("--force-custom-text", action="store_true",
                        help="Force use of CustomTextCLIP model (separate text-tower).")
    parser.add_argument("--context-length", type=int, help="Used to override the text config field `context_length`.")
    parser.add_argument("--torch-script", action="store_true",
                        help="torch.jit.script the model, also uses jit version of OpenAI models if"
                             " pretrained=='openai'")
    parser.add_argument("--torch-compile", action="store_true",
                        help="`torch.compile()` the model. It requires a PyTorch version >= 2.")
    parser.add_argument("--trace", action="store_true", help="torch.jit.trace the model for inference / eval only")
    parser.add_argument("--accum-freq", type=int, default=1, help="Update the model every --accum-freq steps.")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--dist-backend", default="nccl", help="distributed backend")
    parser.add_argument("--report-to", default=[], nargs="*", choices=["neptune", "tensorboard", "wandb"])
    parser.add_argument("--wandb-notes", default="", help="Notes if logging with WandB")
    parser.add_argument("--wandb-project-name", default="open-clip", help="Name of the project if logging with WandB.")
    parser.add_argument("--neptune-project-name", default="bryant1410/open-clip",
                        help="Name of the project if logging with Neptune.")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error", "critical"],
                        help="Logging level.")
    parser.add_argument("--copy-codebase", action="store_true",
                        help="If true, we copy the entire base on the log directory, and execute from there.")
    parser.add_argument("--horovod", action="store_true", help="Use horovod for distributed training.")
    parser.add_argument("--ddp-static-graph", action="store_true", help="Enable static graph optimization for DDP.")
    parser.add_argument("--ddp-gradient-as-bucket-view", action="store_true",
                        help="Enable gradients as a bucket view for DDP.")
    parser.add_argument("--no-set-device-rank", action="store_true",
                        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per"
                             " proc).")
    parser.add_argument("--seed", type=int, default=0, help="Default random seed.")
    parser.add_argument("--grad-clip-norm", type=float, help="Gradient clip.")
    parser.add_argument("--lock-text", action="store_true", help="Lock full text tower by disabling gradients.")
    parser.add_argument("--lock-text-unlocked-layers", type=int, default=0,
                        help="Leave last n text tower layer groups unlocked.")
    parser.add_argument("--lock-text-freeze-layer-norm", action="store_true",
                        help="Freeze LayerNorm running stats in text tower for any locked layers.")
    parser.add_argument("--log-every-n-steps", type=int, default=100,
                        help="Log every n steps to tensorboard/console/wandb.")
    parser.add_argument("--log-first-n-steps", type=int, default=100,
                        help="Log first n steps to tensorboard/console/wandb.")
    parser.add_argument("--loss", default="info_nce",
                        choices=["coca", "info_nce", "siglip"])
    parser.add_argument("--coca-caption-loss-weight", type=float, default=2.0,
                        help="Weight assigned to caption loss in CoCa.")
    parser.add_argument("--coca-contrastive-loss-weight", type=float, default=1.0,
                        help="Weight assigned to contrastive loss when training CoCa.")
    parser.add_argument("--remote-sync", help="Optionally sync with a remote path specified by this arg")
    parser.add_argument("--remote-sync-frequency", type=int, default=60,
                        help="How frequently to sync to a remote directly if --remote-sync is not None.")
    parser.add_argument("--remote-sync-protocol", choices=["s3", "fsspec"], default="s3",
                        help="How to do the remote sync backup if --remote-sync is not None.")
    parser.add_argument("--delete-previous-checkpoint", action="store_true",
                        help="If true, delete previous checkpoint after storing a new one.")
    parser.add_argument("--distill-model", help="Which model arch to distill from, if any.")
    parser.add_argument("--distill-pretrained", help="Which pre-trained weights to distill from, if any.")
    parser.add_argument("--use-bnb-linear",
                        help="Replace the network linear layers from the bitsandbytes library. Allows int8"
                             " training/inference, etc.")
    parser.add_argument("--initial-temperature", type=float,
                        help="Initial temperature for the softmax in the loss. May be overriden by a pre-trained"
                             " model.")
    parser.add_argument("--max-temperature", type=float, default=1.0,
                        help="Maximum temperature for the softmax in the loss.")
    parser.add_argument("--no-fit-temperature", dest="fit_temperature", action="store_false",
                        help="Disable fitting the temperature.")
    parser.add_argument("--sweep-job-id",
                        help="ID to identify a job when doing hparam search, to avoid name clashing.")
    parser.add_argument("--add-extra-caption", action="store_true",
                        help="Load extra captions from the dataset, supposing it's available (e.g., in LAION-COCO).")
    parser.add_argument("--replace-with-extra-caption", action="store_true",
                        help="Replace the caption with the extra caption, instead of adding it.")
    parser.add_argument("--add-random-text-hard-negatives", nargs="?", const="negclip", choices=["negclip", "replace"],
                        help="Add random text hard negatives. When set, by default it's 'negclip'.")
    parser.add_argument("--wise-ft", help="Specify a path to a checkpoint to use for PAINT/WiSE-FT.")
    parser.add_argument("--wise-ft-weight-for-2", type=float, default=0.5, help="Alpha parameter for WiSE-FT, if used.")
    parser.add_argument("--no-halt-on-nan-loss", dest="halt_on_nan_loss", action="store_false",
                        help="Disable halting the program when a NaN loss event occurs.")
    parser.add_argument("--no-log-sys-info", dest="log_sys_info", action="store_false",
                        help="Disable logging the system information, such as the CPU utilization.")
    parser.add_argument("--log-params", action="store_true",
                        help="Log the tune parameters and its grads to TensorBoard.")
    parser.add_argument("--detect-anomalies", action="store_true",
                        help="Uses `torch.set_detect_anomaly`. Use only for debugging as it may slow down the program.")
    return parser


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = create_parser()
    args = parser.parse_args(args)

    for eval_benchmark in args.eval_benchmarks:
        if eval_benchmark not in REGISTERED_EVAL_BENCHMARK and not eval_benchmark.startswith("cb/"):
            raise ValueError(f"Invalid eval benchmark: {eval_benchmark}. Must be one of {REGISTERED_EVAL_BENCHMARK}"
                             f" or start with 'cb'.")

    assert not (args.add_extra_caption and args.add_random_text_hard_negatives), \
        ("`--add-extra-caption` and `--add-random-text-hard-negatives` are mutually exclusive because the code to"
         " support both is not implemented.")

    if not args.eval_batch_size:
        args.eval_batch_size = args.batch_size

    # If some params are not passed, we use the default values based on model name.
    for name, val in _get_default_params(args).items():
        if getattr(args, name) is None:
            setattr(args, name, val)

    if args.train_data:
        args.train_data = "::".join(
            _DATASET_SHORT_NAMES[train_split] if train_split in _DATASET_SHORT_NAMES and not os.path.exists(train_split)
            else train_split
            for train_split in args.train_data.split("::")
        )

    return args
