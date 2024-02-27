import argparse
import logging
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, MutableMapping, Sized
from typing import Any, Literal

import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from webdataset import WebLoader

from open_clip import HFTokenizer, ImageTextModel, get_autocast, get_input_dtype
from open_clip.utils import slugify
from .data import DataInfo
from .metric_utils import compute_image_text_retrieval_metrics
from .utils import argmax_with_random_tiebreaks
from .zero_shot_classifier import build_zero_shot_classifier

_EvalRunner = Callable[[ImageTextModel, Any, argparse.Namespace, HFTokenizer], Mapping[str, float]]

_DATASET_EVAL_REGISTRY: MutableMapping[str, _EvalRunner] = {}


def _eval_dataset(name: str) -> Callable[[_EvalRunner], _EvalRunner]:
    def _wrapper(fn: _EvalRunner) -> _EvalRunner:
        if name in _DATASET_EVAL_REGISTRY:
            raise ValueError(f"An eval runner for {name} is already registered")
        _DATASET_EVAL_REGISTRY[name] = fn
        return fn

    return _wrapper


def _compute_true_positives(scores: torch.Tensor, target: torch.Tensor, top_k: Iterable[int] = (1, 5)) -> Iterator[int]:
    return ((scores.topk(max(top_k), dim=-1)[1] == target.view(-1, 1))[:, :k].sum().item() for k in top_k)


def run_classification_eval(model: ImageTextModel, data: DataLoader, args: argparse.Namespace,
                            tokenizer: HFTokenizer, dataset_name: str | None = None) -> Mapping[str, float]:
    name = dataset_name or getattr(data.dataset, "name", type(data.dataset).__name__)
    class_names = data.dataset.classes  # noqa
    templates = getattr(data.dataset, "templates", ("{}",)) if args.use_templates else ("{}",)

    args_attr_name = f"{slugify(name).replace('-', '_')}_class_text_features"
    if hasattr(args, args_attr_name):
        logging.info(f"Reusing the pre-calculated {name} class representations"
                     f" (because the text tower is frozen)")
        class_text_features = getattr(args, args_attr_name).to(args.device, non_blocking=True)
    else:
        class_text_features = build_zero_shot_classifier(model=model, tokenizer=tokenizer, class_names=class_names,
                                                         args=args, templates=templates, dataset_name=name)

        if not args.text_is_tuned:
            setattr(args, args_attr_name, class_text_features.to("cpu", non_blocking=True))

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    top1, top5, n = 0, 0, 0

    total = len(data.dataset) if isinstance(data.dataset, Sized) else None

    with torch.inference_mode(), \
            tqdm(unit=" images", total=total, desc=f"Running {name} eval") as p_bar:  # noqa
        target_list = []
        pred_list = []

        for batch in data:
            images, target = (batch["image"], batch["target"]) if isinstance(batch, Mapping) else batch

            images = images.to(device=args.device, dtype=input_dtype, non_blocking=True)
            target = target.to(args.device, non_blocking=True)
            target_list.append(target)

            with autocast():
                image_features = model.encode_image(images.reshape(-1, *images.shape[-3:]))

                image_features = F.normalize(image_features.reshape(
                    -1, images.shape[1:-3].numel(), *image_features.shape[1:]).mean(dim=1))

                scores = model.compute_similarity(image_features, class_text_features)

                tp1, tp5 = _compute_true_positives(scores, target)
                top1 += tp1
                top5 += tp5

                batch_size = len(images)
                n += batch_size

                pred_list.append(scores.argmax(dim=-1))

                p_bar.update(batch_size)

        targets = torch.cat(target_list)
        preds = torch.cat(pred_list)

        mean_per_class_recall = balanced_accuracy_score(targets.numpy(force=True), preds.numpy(force=True))

    return {"top1": top1 / n, "top5": top5 / n, "mean_per_class_recall": mean_per_class_recall}


def _maybe_compute_generative_loss(model_out: Mapping[str, torch.Tensor]) -> torch.Tensor | None:
    if "logits" in model_out and "labels" in model_out:
        return F.cross_entropy(model_out["logits"].permute(0, 2, 1), model_out["labels"])
    else:
        return None


def run_retrieval_eval(model: ImageTextModel, data: DataLoader | WebLoader, args: argparse.Namespace,
                       tokenizer: HFTokenizer, dataset_name: str | None = None, **__) -> Mapping[str, float]:
    name = (dataset_name
            or getattr(data, "name", None)
            or getattr(getattr(data, "dataset", None), "name", type(data.dataset).__name__))

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    if hasattr(data, "num_samples"):
        total = data.num_samples
    elif isinstance(data, DataLoader):
        total = len(data.dataset)  # noqa
    else:
        total = None

    with torch.inference_mode(), tqdm(unit=" samples", total=total,
                                      desc=f"Running {name} eval" if name and name != "val" else "Validating") as p_bar:
        all_image_feature_list, all_text_feature_list = [], []

        cumulative_loss = torch.zeros((), device=args.device)
        cumulative_gen_loss = torch.zeros((), device=args.device)
        gen_loss = None

        n = 0

        for batch in data:
            images, texts = (batch["image"], batch["text"]) if isinstance(batch, Mapping) else batch

            if isinstance(texts[0], str):
                texts = tokenizer(texts)

            images = images.to(device=args.device, dtype=input_dtype, non_blocking=True)
            texts = texts.to(device=args.device, non_blocking=True)

            with autocast():
                output = model(images.reshape(-1, *images.shape[-3:]), texts.reshape(-1, *texts.shape[-1:]))

                image_features = output["image_features"]
                text_features = output["text_features"]

                image_features = image_features.reshape(
                    -1, images.shape[1:-3].numel(), *image_features.shape[1:]).mean(dim=1)
                text_features = text_features.reshape(
                    -1, texts.shape[1:-1].numel(), *text_features.shape[1:]).mean(dim=1)

                all_image_feature_list.append(image_features)
                all_text_feature_list.append(text_features)

                logits_per_image = model.compute_similarity(image_features, text_features)
                logits_per_text = model.compute_similarity(text_features, image_features)

                batch_size = len(images)

                target = torch.arange(batch_size, device=images.device).long()
                loss = (F.cross_entropy(logits_per_image, target) +
                        F.cross_entropy(logits_per_text, target)) / 2
                cumulative_loss += loss * batch_size

                gen_loss = _maybe_compute_generative_loss(output)
                if gen_loss is not None:
                    cumulative_gen_loss += gen_loss * batch_size

                n += batch_size

            p_bar.update(batch_size)

    # I believe re-normalizing gives a little boost.
    all_image_features = F.normalize(torch.cat(all_image_feature_list))
    all_text_features = F.normalize(torch.cat(all_text_feature_list))

    metrics = compute_image_text_retrieval_metrics(model, all_image_features, all_text_features)
    metrics = metrics if isinstance(metrics, MutableMapping) else dict(metrics)

    metrics["clip_val_loss"] = (cumulative_loss / n).item()
    metrics["num_samples"] = n

    if gen_loss is not None:
        gen_loss = cumulative_gen_loss / n
        metrics["generative_val_loss"] = gen_loss.item()

    return metrics


def run_clip_benchmark_eval(model: ImageTextModel, data: Any, args: argparse.Namespace,
                            tokenizer: HFTokenizer, dataset_name: str | None = None) -> Mapping[str, float]:
    if hasattr(data.dataset, "classes"):
        return run_classification_eval(model, data, args, tokenizer, dataset_name=dataset_name)
    else:
        return run_retrieval_eval(model, data, args, tokenizer, dataset_name=dataset_name)


def run_zero_shot_eval_one(name: str, model: ImageTextModel, data: Any, args: argparse.Namespace,
                           tokenizer: HFTokenizer) -> Mapping[str, float]:
    if name.startswith("cb/"):
        return run_clip_benchmark_eval(model, data, args, tokenizer, dataset_name=name.removeprefix("cb/"))
    else:
        return _DATASET_EVAL_REGISTRY[name](model, data, args, tokenizer)


@_eval_dataset("val")
def run_val_eval(model: ImageTextModel, data: DataInfo, args: argparse.Namespace,
                 tokenizer: HFTokenizer) -> Mapping[str, float]:
    return run_retrieval_eval(model, data.data_loader, args, tokenizer)


@_eval_dataset("imagenet-val")
def run_imagenet_eval(model: ImageTextModel, data: DataLoader, args: argparse.Namespace,
                      tokenizer: HFTokenizer) -> Mapping[str, float]:
    return run_classification_eval(model, data, args, tokenizer)


@_eval_dataset("imagenet-v2")
def run_imagenet_v2_eval(model: ImageTextModel, data: DataLoader, args: argparse.Namespace,
                         tokenizer: HFTokenizer) -> Mapping[str, float]:
    return run_classification_eval(model, data, args, tokenizer)


@_eval_dataset("winoground")
def run_winoground_eval(model: ImageTextModel, data: DataLoader, args: argparse.Namespace, *_,
                        **__) -> Mapping[str, float]:
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    text_score_pass = 0
    image_score_pass = 0
    group_pass = 0
    total = 0

    with torch.inference_mode(), \
            tqdm(unit=" samples", total=len(data.dataset), desc="Running Winoground eval") as p_bar:  # noqa
        for batch in data:
            image0 = batch["image_0"].to(device=args.device, dtype=input_dtype, non_blocking=True)
            image1 = batch["image_1"].to(device=args.device, dtype=input_dtype, non_blocking=True)
            caption0 = batch["caption_0"].to(device=args.device, non_blocking=True)
            caption1 = batch["caption_1"].to(device=args.device, non_blocking=True)

            with autocast():
                output0 = model(image0, caption0)
                output1 = model(image1, caption1)

                s_t0_i0 = model.compute_similarity_pairwise(output0["text_features"], output0["image_features"])
                s_t0_i1 = model.compute_similarity_pairwise(output0["text_features"], output1["image_features"])
                s_t1_i0 = model.compute_similarity_pairwise(output1["text_features"], output0["image_features"])
                s_t1_i1 = model.compute_similarity_pairwise(output1["text_features"], output1["image_features"])

                batch_text_score_pass = (s_t0_i0 > s_t1_i0) & (s_t1_i1 > s_t0_i1)
                text_score_pass += batch_text_score_pass.sum().item()  # noqa

                batch_image_score_pass = (s_t0_i0 > s_t0_i1) & (s_t1_i1 > s_t1_i0)
                image_score_pass += batch_image_score_pass.sum().item()  # noqa

                batch_group_pass = batch_text_score_pass & batch_image_score_pass
                group_pass += batch_group_pass.sum().item()  # noqa

                total += image0.size(0)

                p_bar.update(image0.size(0))

    return {
        "text-score": text_score_pass / total,
        "image-score": image_score_pass / total,
        "group-score": group_pass / total,
    }


def _run_aro_eval_one(model: ImageTextModel, data_loader: DataLoader, args: argparse.Namespace, correct_index: int = 0,
                      metric_type: Literal["macro", "micro"] = "micro", min_class_count: int = 1, name: str = "",
                      ignore_erroneous_perturbed_captions: bool = True, break_ties_randomly: bool = True) -> float:
    if metric_type not in {"macro", "micro"}:
        raise ValueError(f"`metric_type` must be 'micro' or 'macro', but got {metric_type}")

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    correct = 0 if metric_type == "micro" else defaultdict(int)
    total = 0 if metric_type == "micro" else defaultdict(int)

    with torch.inference_mode(), \
            tqdm(unit=" images", total=len(data_loader.dataset), desc=f"Running ARO/{name} eval") as p_bar:  # noqa
        for batch in data_loader:
            image = batch["image"].to(device=args.device, dtype=input_dtype, non_blocking=True)
            captions = batch["caption_options"].to(device=args.device, non_blocking=True)

            with autocast():
                batch_size, num_text_options, max_text_length = captions.shape

                output = model(image, captions.reshape(-1, max_text_length))
                output["text_features"] = output["text_features"].reshape(batch_size, num_text_options, -1)

                scores = model.compute_similarity(output["image_features"].unsqueeze(1),
                                                  output["text_features"]).squeeze(1)

                if ignore_erroneous_perturbed_captions:
                    # For the "Order" subtasks, ~20% of the examples have a "perturbed" caption that's exactly the same
                    # as the correct caption, which is wrong. Normally, this is not an issue because their scores will
                    # be the same and when you use `argmax`, the first one is going to be picked (and `correct_index` is
                    # always 0). However, we want to avoid selecting the true caption in the cases where there are ties
                    # because the model assigned the same score as a wrong one. So, to separate this case out, we set
                    # the score of these wrong captions as -inf.

                    # Use a range to keep the dimensions.
                    caption_is_same_as_correct = (captions[:, correct_index:correct_index + 1] == captions).all(dim=2)
                    caption_is_same_as_correct[:, correct_index] = False

                    scores[caption_is_same_as_correct] = float("-inf")

                # Without this kind of `argmax`, when we have many ties (e.g., when the model assigns the same embedding
                # to multiple instances, such as when the representations collapse),
                # the results will be inflated.
                # So we perform the fairest thing, which is to break ties randomly when the flag is set.
                argmax_fn = argmax_with_random_tiebreaks if break_ties_randomly else torch.argmax

                batch_correct = argmax_fn(scores, dim=-1) == correct_index

                if metric_type == "micro":
                    correct += batch_correct.sum().item()
                    total += batch_size
                else:
                    for instance_class_name, instance_correct in zip(batch["class_name"], batch_correct):
                        correct[instance_class_name] += instance_correct.item()
                        total[instance_class_name] += 1

            p_bar.update(batch_size)

    if metric_type == "micro":
        return correct / total
    else:
        return sum(correct[k] / total_class
                   for k, total_class in total.items()
                   if total_class >= min_class_count) / sum(total_class >= min_class_count
                                                            for total_class in total.values())


@_eval_dataset("aro")
def run_aro_eval(model: ImageTextModel, data: Mapping[str, DataLoader], args: argparse.Namespace, *_,
                 **__) -> Mapping[str, float]:
    metric_types = {
        "vg-relation": "macro",
        "vg-attribution": "macro",
        "coco-order": "micro",
        "flickr30k-order": "micro",
    }

    min_class_counts = {"vg-relation": 1, "vg-attribution": 25, "coco-order": 1, "flickr30k-order": 1}

    return {
        k: _run_aro_eval_one(model, data_loader, args, metric_type=metric_types[k], min_class_count=min_class_counts[k],
                             name=k.capitalize())
        for k, data_loader in data.items()
    }


@_eval_dataset("color")
def run_color_eval(model: ImageTextModel, data: DataLoader, args: argparse.Namespace,
                   tokenizer: HFTokenizer) -> Mapping[str, float]:
    return run_retrieval_eval(model, data, args, tokenizer, dataset_name="color")


def _run_sugar_crepe_eval_one(model: ImageTextModel, data_loader: DataLoader, args: argparse.Namespace,
                              correct_index: int = 0, name: str = "", break_ties_randomly: bool = True) -> float:
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    correct = 0
    total = 0

    with torch.inference_mode(), tqdm(unit=" samples", total=len(data_loader.dataset),  # noqa
                                      desc=f"Running SugarCrepe/{name} eval") as p_bar:
        for batch in data_loader:
            image = batch["image"].to(device=args.device, dtype=input_dtype, non_blocking=True)
            captions = batch["tested_labels"].to(device=args.device, non_blocking=True)

            with autocast():
                batch_size, num_text_options, max_text_length = captions.shape

                output = model(image, captions.reshape(-1, max_text_length))
                output["text_features"] = output["text_features"].reshape(batch_size, num_text_options, -1)

                scores = model.compute_similarity(output["image_features"].unsqueeze(1),
                                                  output["text_features"]).squeeze(1)

                # Without this kind of `argmax`, when we have many ties (e.g., when the model assigns the same embedding
                # to multiple instances, such as when the representations collapse),
                # the results will be inflated.
                # So we perform the fairest thing, which is to break ties randomly when the flag is set.
                argmax_fn = argmax_with_random_tiebreaks if break_ties_randomly else torch.argmax

                correct += (argmax_fn(scores, dim=-1) == correct_index).sum().item()
                total += batch_size

                p_bar.update(batch_size)

    return correct / total


@_eval_dataset("sugar-crepe")
def run_sugar_crepe_eval(model: ImageTextModel, data: Mapping[str, DataLoader], args: argparse.Namespace, *_,
                         **__) -> Mapping[str, float]:
    return {
        k: _run_sugar_crepe_eval_one(model, data_loader, args, name=k.capitalize())
        for k, data_loader in data.items()
    }


@_eval_dataset("svo-probes")
def run_svo_probes_eval(model: ImageTextModel, data: tuple[DataLoader, DataLoader], args: argparse.Namespace, *_,
                        break_ties_randomly: bool = True, **__) -> Mapping[str, float]:
    image_data_loader, data_loader = data

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    encoded_images = {}

    correct = defaultdict(int)
    total = defaultdict(int)

    with torch.inference_mode():
        with tqdm(unit=" images", total=len(image_data_loader.dataset),  # noqa
                  desc=f"Encoding images for SVO-Probes eval") as p_bar:
            for batch in image_data_loader:
                image = batch["image"].to(device=args.device, dtype=input_dtype, non_blocking=True)

                with autocast():
                    image_features = model.encode_image(image)
                    encoded_images.update((id_instance, image_features_instance)
                                          for id_instance, image_features_instance in zip(batch["id"], image_features))

                p_bar.update(image.shape[0])

        with tqdm(unit=" sentences", total=len(data_loader.dataset), desc=f"Running SVO-Probes eval") as p_bar:  # noqa
            for batch in data_loader:
                sentence = batch["sentence"].to(device=args.device, non_blocking=True)

                with autocast():
                    text_features = model.encode_text(sentence)

                    pos_image_features = torch.stack([encoded_images[id_instance]
                                                      for id_instance in batch["pos_image_id"]])
                    neg_image_features = torch.stack([encoded_images[id_instance]
                                                      for id_instance in batch["neg_image_id"]])

                    pos_scores = model.compute_similarity_pairwise(pos_image_features, text_features)
                    neg_scores = model.compute_similarity_pairwise(neg_image_features, text_features)

                    scores = torch.stack([pos_scores, neg_scores], dim=-1)

                    # Without this kind of `argmax`, when we have many ties
                    # (e.g., when the model assigns the same embedding to multiple instances,
                    # such as when the representations collapse),
                    # the results will be inflated.
                    # So we perform the fairest thing, which is to break ties randomly when the flag is set.
                    argmax_fn = argmax_with_random_tiebreaks if break_ties_randomly else torch.argmax

                    result = argmax_fn(scores, dim=-1) == 0

                    for result_instance, neg_type_instance in zip(result, batch["neg_type"]):
                        correct[neg_type_instance] += result_instance.item()
                        total[neg_type_instance] += 1

                    p_bar.update(sentence.shape[0])

    return {k: correct[k] / total[k] for k in ["s", "v", "o"]}


@_eval_dataset("didemo")
def run_didemo_eval(model: ImageTextModel, data: DataLoader, args: argparse.Namespace,
                    tokenizer: HFTokenizer) -> Mapping[str, float]:
    return run_retrieval_eval(model, data, args, tokenizer, dataset_name="DiDeMo")


@_eval_dataset("hmdb51")
def run_hmdb51_eval(model: ImageTextModel, data: DataLoader, args: argparse.Namespace,
                    tokenizer: HFTokenizer) -> Mapping[str, float]:
    return run_classification_eval(model, data, args, tokenizer)


@_eval_dataset("msrvtt")
def run_msrvtt_eval(model: ImageTextModel, data: DataLoader, args: argparse.Namespace,
                    tokenizer: HFTokenizer) -> Mapping[str, float]:
    return run_retrieval_eval(model, data, args, tokenizer, dataset_name="MSR-VTT")


@_eval_dataset("ucf101")
def run_ucf101_eval(model: ImageTextModel, data: DataLoader, args: argparse.Namespace,
                    tokenizer: HFTokenizer) -> Mapping[str, float]:
    return run_classification_eval(model, data, args, tokenizer)


@_eval_dataset("youcook2")
def run_youcook2_eval(model: ImageTextModel, data: DataLoader, args: argparse.Namespace,
                      tokenizer: HFTokenizer) -> Mapping[str, float]:
    return run_retrieval_eval(model, data, args, tokenizer, dataset_name="YouCook2")


def run_zero_shot_eval(model: ImageTextModel, data: Mapping[str, Any], args: argparse.Namespace,
                       tokenizer: HFTokenizer) -> Mapping[str, float]:
    return {
        (k if name == "val" else f"{name}-{k}"): v
        for name, dataset_data in data.items()
        for k, v in run_zero_shot_eval_one(name, model, dataset_data, args, tokenizer).items()
    }
