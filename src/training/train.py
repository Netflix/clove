import argparse
import json
import logging
import os
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, MutableMapping
from typing import Any, Literal, TypeVar

import math
import psutil
import torch
import wandb
from neptune import Run
from torch import nn
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

from open_clip import ClipLoss, HFTokenizer, ImageTextModel, get_autocast, get_input_dtype, unwrap_model
from .distributed import all_gather_object, is_master
from .metric_utils import compute_image_text_retrieval_metrics, compute_retrieval_metrics
from .scheduler import LrScheduler
from .utils import use_process_no_throw
from .zero_shot import run_zero_shot_eval


# noinspection PyAttributeOutsideInit
class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def _backward(total_loss: torch.Tensor, scaler: GradScaler | None) -> None:
    if scaler is None:
        total_loss.backward()
    else:
        scaler.scale(total_loss).backward()


def _compute_digits(n: int) -> int:
    return math.ceil(math.log10(n + 1))


def train_one_epoch(model: ImageTextModel, data: Mapping[str, Any], loss: nn.Module, epoch: int, optimizer: Optimizer,
                    args: argparse.Namespace, scaler: GradScaler | None = None,
                    scheduler: LrScheduler | None = None, dist_model: ImageTextModel | None = None,
                    tb_writer: SummaryWriter | None = None, neptune_run: Run | None = None) -> Mapping[str, float]:
    device = args.device
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    loss.train()

    if args.distill:
        dist_model.eval()

    data["train"].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch

    epochs = args.epochs
    epochs_digits = _compute_digits(epochs)

    data_loader = data["train"].data_loader

    num_batches = data_multiplier * data_loader.num_batches
    samples_per_epoch = data_multiplier * data_loader.num_samples

    num_batches_per_epoch = num_batches // args.accum_freq
    sample_digits = _compute_digits(samples_per_epoch)

    steps = num_batches_per_epoch * epochs
    steps_digits = _compute_digits(steps)

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], defaultdict(list)
    else:
        accum_images, accum_texts, accum_features = None, None, None

    latest_metrics = {}
    losses_m = defaultdict(AverageMeter)
    contrastive_loss_for_logging = None
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    for i, batch in enumerate(data_loader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        images, texts = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)

        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                model_out = model(images, texts)
                if args.distill:
                    with torch.no_grad():
                        model_out.update((f"dist_{k}", v) for k, v in dist_model(images, texts).items())

                if args.add_random_text_hard_negatives and getattr(loss, "do_gather", True) and args.world_size > 1:
                    # When using text negatives, the shape of the batch is dynamic because sometimes we don't have
                    # negatives.
                    # This sometimes causes an issue with the `all_gather` operation.
                    # It's not clear to me if the `all_gather` officially supports gathering tensors whose first
                    # dimension differs in size. It works fine on a machine with CUDA 11.5 but not on one with CUDA
                    # 11.2 and on Ray (the rest of the high-level things are the same, as far as I can tell).
                    # In any case, the best thing to do seems to pad the batch with extra spurious features.
                    # I think that padding with the null tensor is fine, as the similarity should be 0 and thus it
                    # shouldn't affect much the calculations. Another option could have been to have placeholder
                    # negative inputs, such as the empty text or a random shuffle. The drawback of this would have
                    # been to waste extra compute encoding the empty text, though it maybe has some modeling
                    # advantage?
                    #
                    # Note this error manifests by the gather operation hanging until timeout. It can be debugged
                    # with `TORCH_DISTRIBUTED_DEBUG=DETAIL`, which shows "Detected mismatch between collectives on
                    # ranks."
                    model_out["text_features"] = \
                        torch.cat([model_out["text_features"],
                                   torch.zeros((2 * len(images) - len(texts),
                                                *model_out["text_features"].shape[1:]),
                                               dtype=model_out["text_features"].dtype,
                                               device=model_out["text_features"].device)])

                losses = loss(**model_out)

                total_loss = sum(losses.values())

                if total_loss.isnan() and args.lr_scheduler != "lr-finder":
                    if args.halt_on_nan_loss:
                        raise ValueError("NaN loss encountered")
                    elif is_master(args):
                        print("NaN loss encountered")

                losses["loss"] = total_loss

            _backward(total_loss, scaler)
        else:
            # We don't support when `batch is None` given that counting the number of iterations to accumulate or not
            # gets tricky.
            assert batch is not None, ("Accumulation is only supported without text-only training or with"
                                       " joint-mode text-only training.")

            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, texts)

                    for k, v in model_out.items():
                        if k.endswith("_features"):
                            accum_features[k].append(v)

                accum_images.append(images)
                accum_texts.append(texts)

            if (i + 1) % args.accum_freq > 0:
                continue  # FIXME: this makes data time logging unreliable when accumulating.

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    model_out = model(images, texts)

                    model_out_with_grad = {k: (torch.cat(accum_features[k][:j] + [v] + accum_features[k][j + 1:])
                                               if k in accum_features else v)
                                           for k, v in model_out.items()}
                    losses = loss(**model_out_with_grad)

                    total_loss = sum(losses.values())

                    if total_loss.isnan() and args.lr_scheduler != "lr-finder":
                        if args.halt_on_nan_loss:
                            raise ValueError("NaN loss encountered")
                        elif is_master(args):
                            print("NaN loss encountered")

                    losses["loss"] = total_loss
                _backward(total_loss, scaler)

            accum_images, accum_texts, accum_features = [], [], defaultdict(list)

        if scaler:
            if args.horovod:
                optimizer.synchronize()  # noqa
                scaler.unscale_(optimizer)
                if args.grad_clip_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                with optimizer.skip_synchronize():  # noqa
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()

        with torch.no_grad():  # Note: we clamp to `ln(100)`, as in the original paper.
            unwrap_model(model).logit_scale.clamp_(math.log(1 / args.max_temperature), math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (step <= args.log_first_n_steps
                                or i_accum % args.log_every_n_steps == 0
                                or batch_count == num_batches_per_epoch):
            accum_batch_size = args.accum_freq * args.batch_size
            accum_global_batch_size = accum_batch_size * args.world_size

            samples_per_second = accum_global_batch_size / batch_time_m.val
            samples_per_second_per_gpu = accum_batch_size / batch_time_m.val

            fraction_complete = batch_count / num_batches_per_epoch
            percent_complete = 100.0 * fraction_complete

            num_global_accum_samples = batch_count * accum_global_batch_size

            logit_scale_scalar = model_out["logit_scale"].item()
            logit_bias_scalar = model_out["logit_bias"].item() if "logit_bias" in model_out else None

            if "contrastive_loss" not in losses:
                if contrastive_loss_for_logging is None:
                    # Note we can't gather because only the main node would be running this code.
                    contrastive_loss_for_logging = ClipLoss(model=model, do_gather=False,
                                                            cache_labels=True).to(device)
                with torch.no_grad():
                    losses.update(contrastive_loss_for_logging(**model_out))

            # TODO: pass the gathered features from before, so to compute more accurate stats?
            #  It could only be done if it was done before, not with `contrastive_loss_for_logging`.
            image_text_retrieval_metrics = compute_image_text_retrieval_metrics(
                model, model_out["image_features"].detach(),
                # We consider the case that there are extra negative text features, which we ignore here:
                model_out["text_features"].detach()[:len(model_out["image_features"])])

            # NOTE: the loss is coarsely sampled, just the main node and per log update.
            for k, v in losses.items():
                # FIXME: if we change the text batch size, we need to update here accordingly.
                losses_m[k].update(v.item(), args.batch_size)

            loss_log = " ".join(f"{name.capitalize()}: {m.val:#7.4f} ({m.avg:#7.4f})" for name, m in losses_m.items())

            logging.info(
                f"Train Epoch: {epoch:>{epochs_digits}}/{epochs} "
                + ("" if num_global_accum_samples is None
                   else f"[{num_global_accum_samples:>{sample_digits}}/{samples_per_epoch}"
                        f" ({percent_complete:>3.0f}%)] ") +
                f"Step: {step:>{steps_digits}}/{steps} "
                f"Data (t): {data_time_m.avg:>7.3f} "
                f"Batch (t): {batch_time_m.avg:>7.3f}, {samples_per_second:>5.0f}/s"
                + (f", {samples_per_second_per_gpu:>5.0f}/s/gpu " if args.world_size > 1 else " ") +
                f"LR: {optimizer.param_groups[0]['lr']:.3e} "
                + (f"Logit Scale: {logit_scale_scalar:>7.3f} " if logit_scale_scalar else "")
                + (f"Logit Bias: {logit_bias_scalar:>7.3f} " if logit_bias_scalar else "")
                + loss_log +
                "".join(f" {name}: {image_text_retrieval_metrics[name]:.4f}"
                        for name in ["text_to_image_mean_rank_pct", "text_to_image_mrr"]
                        if name in image_text_retrieval_metrics)
            )

            # Save train loss / etc. Using non-avg meter values as loggers have their own smoothing
            latest_metrics = {
                "train/data_time": data_time_m.val,
                "train/batch_time": batch_time_m.val,
                "train/epoch": epoch + fraction_complete,
                "train/samples_per_second": samples_per_second,
                "train/samples_per_second_per_gpu": samples_per_second_per_gpu,
                "train/lr": optimizer.param_groups[0]["lr"],
                # Only log the losses computed in this step.
                **{f"train/{k}": v.val for k, v in losses_m.items() if k in losses},
                **{f"train/{k}": v for k, v in image_text_retrieval_metrics.items()},
                **{f"train/{k}": v for k, v in text_retrieval_metrics.items()},
            }

            if num_global_accum_samples is not None:
                # FIXME: not that this and other related numbers are spuriously doubled when training with text in
                #  round-robin mode.
                latest_metrics["train/samples_seen"] = samples_per_epoch * epoch + num_global_accum_samples

            if logit_scale_scalar is not None:
                latest_metrics["train/scale"] = logit_scale_scalar

            if logit_bias_scalar is not None:
                latest_metrics["train/bias"] = logit_bias_scalar

            if scaler:
                latest_metrics["train/grad_scale"] = scaler.get_scale()

            latest_metrics["train/weight_norm"] = torch.stack([p.detach().norm()
                                                               for p in model.parameters()]).norm().item()

            latest_metrics["train/grad_norm"] = torch.stack([p.grad.detach().norm()
                                                             for p in model.parameters()
                                                             if p.grad is not None]).norm().item()

            if args.log_sys_info:
                # Use the prefix "sys_" because "sys" clashes with some Neptune naming.

                latest_metrics["sys_/cpu_percent"] = psutil.cpu_percent()

                latest_metrics["sys_/vm_percent"] = psutil.virtual_memory().percent
                latest_metrics["sys_/swap_percent"] = psutil.swap_memory().percent

                latest_metrics["sys_/num_tasks"] = sum(1 for _ in psutil.process_iter())
                latest_metrics["sys_/num_threads"] = sum(use_process_no_throw(p.num_threads)
                                                         for p in psutil.process_iter())

                latest_metrics["sys_/current_process_tree_num_open_fds"] = \
                    sum(use_process_no_throw(p.num_fds) for p in psutil.Process().children(recursive=True))

                latest_metrics.update((f"sys_/cuda_{k}", v) for k, v in torch.cuda.memory_stats(device).items()
                                      if k in {"allocated_bytes.all.current", "allocated_bytes.all.peak",
                                               "num_alloc_retries", "num_ooms", "oversize_allocations.peak",
                                               "reserved_bytes.all.current"})

            for k, v in latest_metrics.items():
                if tb_writer:
                    tb_writer.add_scalar(k, v, step)
                if neptune_run and math.isfinite(v):  # Otherwise, it stops tracking with Neptune.
                    neptune_run[k].append(v, step=step)

            if args.wandb:
                assert wandb, "Please install wandb."
                wandb.log({"step": step, **latest_metrics}, step=step)

            if tb_writer and args.log_params:
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        if param.grad is not None:
                            tb_writer.add_histogram(f"grad/{name}", param.grad.detach().flatten(), step)
                        tb_writer.add_histogram(f"param/{name}", param.detach().flatten(), step)

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()

    return latest_metrics


def evaluate(model: ImageTextModel | DistributedDataParallel, data: Mapping[str, Any], epoch: int,
             args: argparse.Namespace, tokenizer: HFTokenizer, tb_writer: SummaryWriter | None = None,
             neptune_run: Run | None = None) -> Mapping[str, float]:
    data = {
        k: v
        for i, (k, v) in enumerate(data.items())
        if (
                epoch == args.epochs
                or (k == "val" and args.val_frequency and epoch % args.val_frequency == 0)
                or (k != "val" and args.zeroshot_frequency and epoch % args.zeroshot_frequency == 0)
        )
    }

    model.eval()
    model = unwrap_model(model)

    try:
        metrics = run_zero_shot_eval(model, data, args, tokenizer)
    except KeyboardInterrupt:
        raise
    except:  # noqa
        metrics = {}
        logging.error(f"Failed to run the zero-shot eval for {list(data.keys())}. See the error below", exc_info=1)

    metrics = metrics if isinstance(metrics, MutableMapping) else dict(metrics)

    if args.world_size > 1:
        metrics = {k: v for rank_metrics in all_gather_object(args, metrics) for k, v in rank_metrics.items()}

    # Note we can't return earlier (e.g., if `data` is empty for the current rank) because we need to gather all first.
    if not is_master(args) or not metrics:
        return {}

    metrics["epoch"] = epoch

    logging.info(f"Eval Epoch: {epoch} " + "\t".join(f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()))

    metrics = {f"val/{name}": val for name, val in metrics.items()}

    if args.save_logs:
        if tb_writer:
            for name, val in metrics.items():
                tb_writer.add_scalar(name, val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb, "Please install wandb."

        if "train" in data:
            num_batches_per_epoch = data["train"].dataloader.num_batches // args.accum_freq
            step = num_batches_per_epoch * epoch
        else:
            step = None

        wandb.log({"epoch": epoch, **metrics}, step=step)

    if neptune_run:
        for name, val in metrics.items():
            neptune_run[name].append(val, step=epoch)

    return metrics
