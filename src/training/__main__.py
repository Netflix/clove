import os

CACHE_DIR = os.getenv("CACHE_DIR", "~/.cache")
os.environ.setdefault("CACHE_DIR", CACHE_DIR)

hf_cache_dir = os.path.join(CACHE_DIR, "huggingface")
os.makedirs(CACHE_DIR, exist_ok=True)

if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = hf_cache_dir

if "HF_CACHE_HOME" not in os.environ:
    os.environ["HF_CACHE_HOME"] = hf_cache_dir

import argparse
import glob
import json
import logging
import os
import random
import re
import subprocess
import sys
from datetime import datetime
from typing import Any, TYPE_CHECKING
from collections.abc import Iterable, Mapping

import numpy as np
import torch
import wandb
from torch import optim
from torch.cuda.amp import GradScaler
import torch.utils.tensorboard as tensorboard

from open_clip import create_loss, create_model, create_model_and_preprocessing, trace_model, unwrap_model

from training.data import get_data
from training.distributed import broadcast_object, init_distributed_device, is_master
from training.file_utils import pt_load, remote_sync, start_sync_process
from training.logger import setup_logging
from training.params import parse_args
from training.scheduler import const_lr, const_lr_cooldown, cosine_lr, lr_finder
from training.train import evaluate, train_one_epoch
from training.utils import get_state_dict, patch_model

try:
    import horovod.torch as hvd
except ImportError:
    hvd = Any if TYPE_CHECKING else None

LATEST_CHECKPOINT_NAME = "epoch_latest.pt"


def _random_seed(seed: int = 42, rank: int = 0) -> None:
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def _natural_key(string_: str) -> tuple[int, ...]:
    """See https://blog.codinghorror.com/sorting-for-humans-natural-sort-order/."""
    return tuple(int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_.lower()))


def _get_latest_checkpoint(path: str, remote: bool) -> str | None:
    # As written, this glob is recursive, so can pick up checkpoints across multiple subdirectories.
    if remote:
        result = subprocess.run(["aws", "s3", "ls", path.rstrip("/") + "/"], capture_output=True)
        logging.info(result)
        if result.returncode == 1:
            return None
        checkpoints = (os.path.join(path, x.split(" ")[-1]) for x in result.stdout.decode().split("\n")[:-1])
    else:
        checkpoints = glob.iglob(path + "**/*.pt", recursive=True)

    return max(checkpoints, key=_natural_key, default=None)


def _copy_codebase(args: argparse.Namespace) -> str | None:
    new_code_path = os.path.join(args.logs, args.name, "code")
    if os.path.exists(new_code_path):
        logging.error(f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment.")
        return None
    logging.info(f"Copying codebase to {new_code_path}")
    current_code_path = os.path.realpath(__file__)

    import gitignorefile
    git_ignore_fn = gitignorefile.ignore()

    from shutil import copytree

    if os.path.basename(os.path.dirname(os.path.dirname(current_code_path))).startswith("_ray_pkg_"):
        # We're running on a Ray cluster, which means that we just have each module in a directory.

        copytree(os.path.dirname(current_code_path), os.path.join(new_code_path, "training"), ignore=git_ignore_fn)

        import open_clip
        copytree(os.path.dirname(os.path.realpath(open_clip.__file__)), os.path.join(new_code_path, "open_clip"),
                 ignore=git_ignore_fn)
    else:
        for _ in range(3):
            current_code_path = os.path.dirname(current_code_path)

        copytree(current_code_path, new_code_path, ignore=lambda root, names: git_ignore_fn(root, names) | {".git"})

        if os.path.exists(os.path.join(current_code_path, ".git")):
            with open(os.path.join(new_code_path, ".last_git_commit"), "w") as file:
                try:
                    command_output = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                                    check=True, text=True)
                    file.write(command_output.stdout)
                except subprocess.CalledProcessError:
                    logging.info("Error getting the latest Git commit hash to save it with the codebase copy:",
                                 command_output.stderr, exc_info=1)

    logging.info("Done copying code.")

    return new_code_path


def main(args: Iterable[str] | None = None) -> Mapping[str, float]:
    args = parse_args(args)

    # Experimentally, I found that using `OMP_NUM_THREADS=1` was too slow with a batch size of 1024 and 24 data loader
    # workers (PT 1.13). As soon as I unset this variable, it was much faster. In accordance with this, [a GitHub
    # comment](https://github.com/pytorch/pytorch/issues/22260#issuecomment-508196387) says that the value should be
    # equal to the number of CPU cores assigned per running process.
    # As a side note, contradictorily, an internal doc says the opposite: [REDACTED]
    # Maybe it's because the recommendation applies only to CPU inference in PyTorch?
    os.environ.setdefault("OMP_NUM_THREADS", str(max(args.workers, 1)))

    if "CACHED_PATH_CACHE_ROOT" not in os.environ:
        cached_path_dir = os.path.join(CACHE_DIR, "cached_path")
        os.makedirs(cached_path_dir, exist_ok=True)

        import cached_path
        cached_path.set_cache_dir(cached_path_dir)

    if torch.cuda.is_available():
        # This enables TF32 on Ampere GPUs, which is only 8% slower than float16 and almost as accurate as float32.
        # This was a default in pytorch until 1.12.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # fully initialize distributed device environment
    device = init_distributed_device(args)

    # get the name of the experiments
    if args.name is None:
        # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
        model_name_safe = args.model.replace("/", "-")
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S_%f")
        if args.distributed:
            # Sync this object from the main node to all ranks.
            date_str = broadcast_object(args, date_str)

        name_args = [
            date_str,
            f"model_{model_name_safe}",
            f"opt_{args.optimizer}",
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
            f"j_{args.workers}",
            f"p_{args.precision}",
        ]

        if args.sweep_job_id is not None:
            name_args.append(f"s_{args.sweep_job_id}")

        args.name = "-".join(name_args)

    resume_latest = args.resume == "latest"
    log_base_path = os.path.join(args.logs, args.name)
    args.log_path = None
    if is_master(args, local=args.log_local):
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f"out-{args.rank}" if args.log_local else "out.log"
        args.log_path = os.path.join(log_base_path, log_filename)
        if os.path.exists(args.log_path) and not resume_latest:
            print(  # We use `print` because the logging setup hasn't happened yet.
                "Error. Experiment already exists. Use --name {} to specify a new experiment."
            )
            return {}

    setup_logging(logging.getLevelName(args.log_level.upper()), args.log_path)

    logging.info(f"Input args: {sys.argv}")

    logging.info(f"OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']}")

    ray_job_config_json = os.environ.get("RAY_JOB_CONFIG_JSON_ENV_VAR")
    if ray_job_config_json:
        ray_job_config = json.loads(ray_job_config_json)
        ray_job_submission_id = ray_job_config.get("metadata", {}).get("job_submission_id")
        if ray_job_submission_id:
            logging.info(f"Ray job submission ID: {ray_job_submission_id} (from RAY_JOB_CONFIG_JSON_ENV_VAR)")
        else:
            logging.info("RAY_JOB_CONFIG_JSON_ENV_VAR is set but `job_submission_id` couldn't be parsed.")
    else:
        logging.info("RAY_JOB_CONFIG_JSON_ENV_VAR is unset.")

    # Setup wandb, tensorboard, checkpoint logging
    args.neptune = "neptune" in args.report_to
    args.tensorboard = "tensorboard" in args.report_to
    args.wandb = "wandb" in args.report_to
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    if is_master(args):
        args.tensorboard_path = os.path.join(log_base_path, "tensorboard") if args.tensorboard else ""
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ""

    if resume_latest:
        resume_from = None
        checkpoint_path = args.checkpoint_path
        # If using remote_sync, need to check the remote instead of the local checkpoints' directory.
        if args.remote_sync is not None:
            checkpoint_path = os.path.join(args.remote_sync, args.name, "checkpoints")
            if args.save_most_recent:
                logging.error("Cannot use save-most-recent with remote_sync and resume latest.")
                return {}
            if args.remote_sync_protocol != "s3":
                logging.error("Sync protocol not supported when using resume latest.")
                return {}
        if is_master(args):
            # Checking for existing checkpoint via the main rank only.
            # It is possible for different rank processes to see different files if a shared file-system is under
            # stress.
            # However, it's really hard to fully work around such situations.
            if args.save_most_recent:
                # If --save-most-recent flag is set, look for the latest at a fixed filename.
                resume_from = os.path.join(checkpoint_path, LATEST_CHECKPOINT_NAME)
                if not os.path.exists(resume_from):
                    # If no latest checkpoint has been saved yet, don't try to resume
                    resume_from = None
            else:
                # otherwise, list checkpoint dir contents and pick the newest checkpoint
                resume_from = _get_latest_checkpoint(checkpoint_path, remote=args.remote_sync is not None)
            if resume_from:
                logging.info(f"Found latest resume checkpoint at {resume_from}.")
            else:
                logging.info(f"No latest resume checkpoint found in {checkpoint_path}.")
        if args.distributed:
            # sync found checkpoint path to all ranks
            resume_from = broadcast_object(args, resume_from)
        args.resume = resume_from

    codebase_copy_path = _copy_codebase(args) if is_master(args) and args.copy_codebase else None

    # Start the sync process if remote-syncing.
    remote_sync_process = None
    if is_master(args) and args.remote_sync:
        # First, let's make sure it works.
        result = remote_sync(
            os.path.join(args.logs, args.name),
            os.path.join(args.remote_sync, args.name),
            args.remote_sync_protocol
        )
        if result:
            logging.info("Initial remote sync successful.")
        else:
            logging.info("Error: initial remote sync failed. Exiting.")
            return {}
        # If it's all looks good, start a process to do this every `args.remote_sync_frequency` seconds.
        remote_sync_process = start_sync_process(
            args.remote_sync_frequency,
            os.path.join(args.logs, args.name),
            os.path.join(args.remote_sync, args.name),
            args.remote_sync_protocol
        )
        remote_sync_process.start()

    if args.precision == "fp16":
        logging.warning(
            "It is recommended to use AMP mixed-precision instead of FP16. "
            "FP16 support needs further verification and tuning, especially for train.")

    if args.horovod:
        logging.info(
            f"Running in horovod mode with multiple processes / nodes. Device: {args.device}."
            f"Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.")
    elif args.distributed:
        logging.info(
            f"Running in distributed mode with multiple processes. Device: {args.device}."
            f"Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.")
    else:
        logging.info(f"Running with a single process. Device {args.device}.")

    args.distill = args.distill_model and args.distill_pretrained
    if args.distill:
        # FIXME: support distillation with grad accum.
        assert args.accum_freq == 1
        # FIXME: support distillation with coca.
        assert "coca" not in args.model.lower()
        assert args.loss not in {"barlow_twins", "corinfomax", "mse"}
    elif args.loss in {"barlow_twins", "corinfomax", "mse"}:
        assert "coca" not in args.model.lower()

    if isinstance(args.force_image_size, (tuple, list)) and len(args.force_image_size) == 1:
        # arg is nargs, single (square) image size list -> int
        args.force_image_size = args.force_image_size[0]
    _random_seed(args.seed, rank=0)

    model_kwargs = {}

    if args.loss == "siglip":
        model_kwargs["init_logit_bias"] = -10

    model, train_transform, val_transform, tokenizer = create_model_and_preprocessing(
        args.model,
        args.pretrained,
        context_length=args.context_length,
        precision=args.precision,
        device=device,
        jit=args.torch_script,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_patch_dropout=args.force_patch_dropout,
        pretrained_image=args.pretrained_image,
        image_mean=args.image_mean,
        image_std=args.image_std,
        image_interpolation=args.image_interpolation,
        image_resize_mode=args.image_resize_mode,  # Only effective for inference.
        aug_cfg=args.aug_cfg,
        force_image_size=args.force_image_size,
        initial_temperature=args.initial_temperature,
        fit_temperature=args.fit_temperature,
        **model_kwargs,
    )

    args.embed_dim = model.embed_dim

    # FIXME: currently assumes the model you're distilling from has the same tokenizer, transforms & init temp.
    dist_model = create_model(
        args.distill_model,
        args.distill_pretrained,
        device=device,
        precision=args.precision,
        initial_temperature=args.initial_temperature,
        fit_temperature=args.fit_temperature,
    ) if args.distill else None

    if args.use_bnb_linear:
        logging.info("=> using a layer from bitsandbytes.\n"
                     "   this is an experimental feature which requires two extra pip installs:\n"
                     "   pip install bitsandbytes triton\n"
                     "   Please make sure to use triton 2.0.0.")
        import bitsandbytes as bnb
        from open_clip.utils import replace_linear
        logging.info(f"=> replacing linear layers with {args.use_bnb_linear}")
        linear_replacement_cls = getattr(bnb.nn.triton_based_modules, args.use_bnb_linear)
        replace_linear(model, linear_replacement_cls)
        model = model.to(device)

    _random_seed(args.seed, args.rank)

    if args.trace:
        model = trace_model(model, batch_size=args.batch_size, device=device)

    if args.lock_image:
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        model.lock_image_tower(unlocked_groups=args.lock_image_unlocked_groups,
                               freeze_bn_stats=args.lock_image_freeze_bn_stats)
    if args.lock_text:
        model.lock_text_tower(unlocked_layers=args.lock_text_unlocked_layers,
                              freeze_layer_norm=args.lock_text_freeze_layer_norm)

    if args.grad_checkpointing:
        model.set_grad_checkpointing()

    if is_master(args):
        logging.info("Model:")
        logging.info(model)
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")
    else:
        params_file = None

    if args.distributed and not args.horovod:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_args = {}
        if args.ddp_static_graph:
            ddp_args["static_graph"] = True
        if args.ddp_gradient_as_bucket_view:
            ddp_args["gradient_as_bucket_view"] = True
        # TODO: if we don't use `args.local_loss`, do we need to sync the grads at all? Because each process is
        #  going to materialize the exact same matrix and syncing the grads at that point. So it sounds like the DDP is
        #  going to do an unnecessary extra grad sync (for all params). Maybe we can save this data transfer overhead by
        #  not using DDP at all? Supposing that DDP isn't clever enough to save it automatically. Though maybe DDP is
        #  necessary for something else?
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)

        if args.distill:
            dist_model = torch.nn.parallel.DistributedDataParallel(dist_model, device_ids=[device], **ddp_args)

    # create optimizer and scaler
    optimizer = None
    scaler = None

    if args.train_data or args.dataset_type == "synthetic":
        assert not args.trace, "Cannot train with traced model"

        exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or "logit_scale" in n
        include = lambda n, p: not exclude(n, p)

        named_parameters = list(model.named_parameters())
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

        params = [
            {"params": gain_or_bias_params, "weight_decay": 0.},
            {"params": rest_params, "weight_decay": args.wd},
        ]

        if args.optimizer == "adamw":
            optimizer = optim.AdamW(params, lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps)
        elif args.optimizer == "sgd":
            optimizer = optim.SGD(params, lr=args.lr)
        else:
            raise ValueError(f"Unknown optimizer: '{args.optimizer}'")

        if args.horovod:
            optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        scaler = GradScaler() if args.precision == "amp" else None

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume:
        checkpoint = pt_load(args.resume)

        model.load_state_dict(get_state_dict(checkpoint, model=model, is_distributed=args.distributed))

        if "epoch" in checkpoint:  # Resuming a train checkpoint w/ epoch and optimizer state.
            start_epoch = checkpoint["epoch"]

            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])

            if scaler is not None and "scaler" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler"])

            logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
        else:  # Loading a bare (model only) checkpoint for fine-tune or evaluation.
            logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")

    if args.wise_ft:
        patch_model(model, get_state_dict(pt_load(args.wise_ft), model=model, is_distributed=args.distributed),
                    weight_for_state_dict=args.wise_ft_weight_for_2)
        logging.info(f"=> loaded checkpoint '{args.wise_ft}' for WiSE-FT")

    data = get_data(args=args, train_transform=train_transform, val_transform=val_transform, tokenizer=tokenizer,
                    epoch=start_epoch)
    if not data:
        # We don't want to raise an exception here because we may be doing an evaluation in parallel,
        # and there may be fewer datasets to evaluate than GPUs.
        logging.info("No data found. Exiting.")
        return {}

    scheduler = None
    if "train" in data and optimizer is not None:
        total_steps = (data["train"].data_loader.num_batches // args.accum_freq) * args.epochs
        if args.lr_scheduler == "cosine":
            scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const":
            scheduler = const_lr(optimizer, args.lr, args.warmup)
        elif args.lr_scheduler == "const-cooldown":
            assert args.epochs_cooldown is not None, \
                "Please specify the number of cooldown epochs for this LR schedule."
            cooldown_steps = (data["train"].data_loader.num_batches // args.accum_freq) * args.epochs_cooldown
            scheduler = const_lr_cooldown(optimizer, args.lr, args.warmup, total_steps, cooldown_steps,
                                          args.lr_cooldown_power, args.lr_cooldown_end)
        elif args.lr_scheduler == "lr-finder":
            scheduler = lr_finder(optimizer, min_lr=args.lr, max_lr=10, steps=total_steps)
        else:
            raise ValueError(f"Unknown LR scheduler: {args.lr_scheduler}.")

    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = args.logs and args.logs.lower() != "none" and is_master(args)
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    if args.wandb and is_master(args):
        assert wandb is not None, "Please install wandb."
        logging.debug("Starting WandB.")
        args.train_sz = data["train"].data_loader.num_samples
        if "val" in data:
            args.val_sz = data["val"].data_loader.num_samples
        # you will have to configure this for your project!
        wandb.init(
            project=args.wandb_project_name,
            name=args.name,
            id=args.name,
            notes=args.wandb_notes,
            tags=[],
            resume="auto" if args.resume == "latest" else None,
            config=vars(args),
        )
        if args.log_level == "debug":
            wandb.watch(model, log="all")
        wandb.save(params_file)
        logging.debug("Finished loading WandB.")

    if args.neptune and is_master(args):
        logging.debug("Starting Neptune.")

        import neptune.utils

        source_files = os.path.join(codebase_copy_path, "**") if codebase_copy_path else None

        neptune_run = neptune.init_run(project=args.neptune_project_name, source_files=source_files,
                                       dependencies="requirements.txt")
        logging.info(f"Logging with Neptune: {neptune_run.get_url()}")

        neptune_run["hyperparams"] = neptune.utils.stringify_unsupported(vars(args))

        logging.debug("Finished loading Neptune.")
    else:
        neptune_run = None

    # PyTorch 2.0 adds the "_orig_mod." prefix to keys of state_dict() of compiled models.
    # For compatibility, we save state_dict() of the original model, which shares the weights without the prefix.
    original_model = model
    if args.torch_compile:
        logging.info("Compiling model...")
        model = torch.compile(original_model)

    model_to_check = unwrap_model(model)
    if hasattr(model_to_check, "text"):
        submodules_to_check = ["text"]
        parameters_to_check = []
    elif hasattr(model_to_check, "transformer"):
        submodules_to_check = ["transformer", "token_embedding", "ln_final"]
        parameters_to_check = ["positional_embedding", "text_projection"]
    elif hasattr(model_to_check, "textual"):
        submodules_to_check = ["textual", "textual_proj"]
        parameters_to_check = []
    else:
        raise ValueError("Unknown model type when inferring whether the text tower is being tuned.")
    args.text_is_tuned = any(p.requires_grad
                             for m in submodules_to_check
                             for p in getattr(model_to_check, m).parameters()
                             ) or any(p.requires_grad
                                      for name in parameters_to_check
                                      for p in getattr(model_to_check, name))

    if "train" not in data:
        if args.use_bnb_linear:  # If using int8, convert to inference mode.
            from open_clip.utils import convert_int8_model_to_inference_mode
            convert_int8_model_to_inference_mode(model)

        return evaluate(model, data, start_epoch, args, tokenizer, tb_writer=writer, neptune_run=neptune_run)

    if is_master(args):
        logging.info(f"Frozen params: {[n for n, p in model.named_parameters() if not p.requires_grad]}")
        logging.info(f"Tuned params: {[n for n, p in model.named_parameters() if p.requires_grad]}")

        logging.info(f"Tuned param count: {sum(p.numel() for p in model.parameters() if p.requires_grad):.1e}")

    if args.pretrained and any(not v.startswith("train") for v in data.keys()):
        # Note that, if `args.context_length` is set, because randomly initialized position embeddings may be added,
        # the evaluation can be different from the vanilla pre-trained model.
        latest_metrics = evaluate(model, data, start_epoch, args, tokenizer, tb_writer=writer, neptune_run=neptune_run)
    else:
        latest_metrics = {}

    loss = create_loss(args, model)
    loss = loss.to(device)

    if args.detect_anomalies:
        logging.info("Enabling anomaly detection.")
        torch.set_anomaly_enabled(True)

    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            logging.info(f"Start epoch {epoch}")

        latest_metrics = train_one_epoch(model, data, loss, epoch, optimizer, args, scaler=scaler, scheduler=scheduler,
                                         dist_model=dist_model, tb_writer=writer, neptune_run=neptune_run)
        completed_epoch = epoch + 1

        if any(not v.startswith("train") for v in data.keys()):
            latest_metrics = {**latest_metrics, **evaluate(model, data, completed_epoch, args, tokenizer,
                                                           tb_writer=writer, neptune_run=neptune_run)}

        # Saving checkpoints.
        if args.save_logs:
            checkpoint_dict = {
                "epoch": completed_epoch,
                "name": args.name,
                "state_dict": original_model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if scaler is not None:
                checkpoint_dict["scaler"] = scaler.state_dict()

            if completed_epoch == args.epochs or (args.save_frequency > 0
                                                  and (completed_epoch % args.save_frequency) == 0):
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
                )
            if args.delete_previous_checkpoint:
                previous_checkpoint = os.path.join(args.checkpoint_path, f"epoch_{completed_epoch - 1}.pt")
                if os.path.exists(previous_checkpoint):
                    os.remove(previous_checkpoint)

            if args.save_most_recent:
                # try not to corrupt the latest checkpoint if save fails
                tmp_save_path = os.path.join(args.checkpoint_path, "tmp.pt")
                latest_save_path = os.path.join(args.checkpoint_path, LATEST_CHECKPOINT_NAME)
                torch.save(checkpoint_dict, tmp_save_path)
                os.replace(tmp_save_path, latest_save_path)

    if args.wandb and is_master(args):
        wandb.finish()

    if args.neptune and is_master(args):
        neptune_run.stop()

    # run a final sync.
    if remote_sync_process is not None:
        logging.info("Final remote sync.")
        remote_sync_process.terminate()
        result = remote_sync(
            os.path.join(args.logs, args.name),
            os.path.join(args.remote_sync, args.name),
            args.remote_sync_protocol
        )
        if result:
            logging.info("Final remote sync successful.")
        else:
            logging.info("Final remote sync failed.")

    return latest_metrics


if __name__ == "__main__":
    main()
