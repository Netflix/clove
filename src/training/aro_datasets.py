# Originally copied from https://github.com/mertyg/vision-language-models-are-bows/tree/81fdd20/dataset_zoo
import json
import logging
import os
import random
import re
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping, MutableMapping, Sequence
from typing import Any, Literal

import gdown
import nltk
import numpy as np
import spacy.tokens
from PIL import Image
from filelock import FileLock
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from tqdm import tqdm

from open_clip.transform import Transform
from open_clip.utils import maybe_nltk_download
from training.s3_utils import s3_sync

CACHE_DIR = os.getenv("CACHE_DIR", "/mnt/.cache")

NLP = spacy.load("en_core_web_sm", enable=["tok2vec", "tagger"])  # If "tok2vec" is not specified, all tags are NN.


def _get_spacy_docs(texts: Iterable[str]) -> Iterator[spacy.tokens.Doc]:
    # I have tested it with `n_process>1` but it was considerably slower.
    # See https://github.com/explosion/spaCy/discussions/10306#discussioncomment-2227033
    return NLP.pipe(texts, batch_size=2048)


def shuffle_nouns_and_adj(text: str, spacy_doc: spacy.tokens.Doc) -> str:  # noqa
    text = np.array([token.text for token in spacy_doc])
    noun_idx = [i for i, token in enumerate(spacy_doc) if token.tag_ in {"NN", "NNS", "NNP", "NNPS"}]
    adjective_idx = [i for i, token in enumerate(spacy_doc) if token.tag_ in {"JJ", "JJR", "JJS"}]
    text[noun_idx] = np.random.permutation(text[noun_idx])  # Shuffle the nouns of the text
    text[adjective_idx] = np.random.permutation(text[adjective_idx])  # Shuffle the adjectives of the text
    return " ".join(text)


def shuffle_all_but_nouns_and_adj(text: str, spacy_doc: spacy.tokens.Doc) -> str:  # noqa
    text = np.array([token.text for token in spacy_doc])
    noun_adj_idx = [i for i, token in enumerate(spacy_doc) if
                    token.tag_ in {"NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS"}]

    else_idx = np.ones(text.shape[0])
    else_idx[noun_adj_idx] = 0
    else_idx = else_idx.astype(bool)
    text[else_idx] = np.random.permutation(text[else_idx])

    return " ".join(text)


# Taken from:
# https://github.com/lingo-mit/context-ablations/blob/478fb18/code/transformers/src/transformers/data/data_augmentation.py
def _get_trigrams(sentence: Iterable[str]) -> list[list[str]]:
    trigrams = []
    trigram = []
    for i, s in enumerate(sentence):
        trigram.append(s)
        if i % 3 == 2:
            trigrams.append(trigram[:])
            trigram = []
    if trigram:
        trigrams.append(trigram)
    return trigrams


def _trigram_shuffle(sentence: Iterable[str]) -> str:
    trigrams = _get_trigrams(sentence)
    for trigram in trigrams:
        random.shuffle(trigram)
    return " ".join(" ".join(trigram) for trigram in trigrams)


def shuffle_within_trigrams(text: str, spacy_doc: spacy.tokens.Doc) -> str:  # noqa
    return _trigram_shuffle(nltk.word_tokenize(text))


def shuffle_trigrams(text: str, spacy_doc: spacy.tokens.Doc) -> str:  # noqa
    trigrams = _get_trigrams(nltk.word_tokenize(text))
    random.shuffle(trigrams)
    return " ".join(" ".join(trigram) for trigram in trigrams)


def pre_caption(caption: str, max_words: int = 50) -> str:
    caption = re.sub(r"([.!\"()*#:;~])", " ", caption.lower())
    caption = re.sub(r"\s{2,}", " ", caption)
    caption = caption.rstrip("\n")
    caption = caption.strip(" ")

    caption_words = caption.split(" ")
    if len(caption_words) > max_words:
        caption = " ".join(caption_words[:max_words])

    return caption


class AroDataset(Dataset, ABC):
    def __init__(self, instances: Sequence[MutableMapping[str, Any]], root_dir: str, transform: Transform) -> None:
        self.instances = instances
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, i: int) -> Mapping[str, Any]:
        instance = self.instances[i]
        image = Image.open(os.path.join(self.root_dir, instance["image_path"])).convert("RGB")

        if "bbox_x" in instance:
            # Get the bounding box that contains the relation. This is to remove the irrelevant details in the scene.
            image = image.crop((instance["bbox_x"], instance["bbox_y"], instance["bbox_x"] + instance["bbox_w"],
                                instance["bbox_y"] + instance["bbox_h"]))

        output = {"image": self.transform(image), "caption_options": instance["caption_options"]}

        if "class_name" in instance:
            output["class_name"] = instance["class_name"]

        return output


class AroVgDataset(AroDataset, ABC):
    DEFAULT_ROOT_DIR = os.getenv("ARO_VG_ROOT_DIR", os.path.join(CACHE_DIR, "prerelease_bow"))

    def __init__(self, transform: Transform, annotation_filename: str, annotation_file_google_drive_id: str,
                 root_dir: str = DEFAULT_ROOT_DIR, download: bool = True) -> None:
        annotation_file = os.path.join(root_dir, annotation_filename)
        image_dir = os.path.join(root_dir, "images")
        if not os.path.exists(image_dir):
            if download:
                os.makedirs(os.path.dirname(image_dir), exist_ok=True)
                with FileLock(image_dir + ".lock"):
                    if not os.path.exists(image_dir):  # Retry in case the process had been locked.
                        logging.info("Downloading ARO-VG dataset.")
                        os.makedirs(root_dir, exist_ok=True)
                        filename = "vgr_vga_images.zip"
                        gdown.download(id="1qaPlrwhGNMrR3a11iopZUT_GPP_LrgP9",  # noqa
                                       output=os.path.join(root_dir, filename), use_cookies=False)
                        subprocess.run(["unzip", filename], cwd=root_dir, check=True)
            else:
                raise RuntimeError("Dataset not found.")

        if not os.path.exists(annotation_file):
            with FileLock(annotation_file + ".lock"):
                if not os.path.exists(annotation_file):  # Retry in case the process had been locked.
                    gdown.download(id=annotation_file_google_drive_id, output=annotation_file)

        with open(annotation_file) as f:
            instances = json.load(f)

        for instance in instances:
            instance["caption_options"] = [instance["true_caption"], instance["false_caption"]]

        super().__init__(instances, image_dir, transform)


class AroVgRelation(AroVgDataset):
    SYMMETRIC_RELATIONSHIPS = {
        "adjusting", "attached to", "between", "bigger than", "biting", "boarding", "brushing", "chewing", "cleaning",
        "climbing", "close to", "coming from", "coming out of", "contain", "crossing", "dragging", "draped over",
        "drinking", "drinking from", "driving", "driving down", "driving on", "eating from", "eating in", "enclosing",
        "exiting", "facing", "filled with", "floating in", "floating on", "flying", "flying above", "flying in",
        "flying over", "flying through", "full of", "going down", "going into", "going through", "grazing in",
        "growing in", "growing on", "guiding", "hanging from", "hanging in", "hanging off", "hanging over",
        "higher than", "holding onto", "hugging", "in between", "jumping off", "jumping on", "jumping over",
        "kept in", "larger than", "leading", "leaning over", "leaving", "licking", "longer than", "looking in",
        "looking into", "looking out", "looking over", "looking through", "lying next to", "lying on top of", "making",
        "mixed with", "mounted on", "moving", "on the back of", "on the edge of", "on the front of",
        "on the other side of", "opening", "painted on", "parked at", "parked beside", "parked by", "parked in",
        "parked in front of", "parked near", "parked next to", "perched on", "petting", "piled on", "playing",
        "playing in", "playing on", "playing with", "pouring", "reaching for", "reading", "reflected on", "riding on",
        "running in", "running on", "running through", "seen through", "sitting behind", "sitting beside",
        "sitting by", "sitting in front of", "sitting near", "sitting next to", "sitting under", "skiing down",
        "skiing on", "sleeping in", "sleeping on", "smiling at", "sniffing", "splashing", "sprinkled on", "stacked on",
        "standing against", "standing around", "standing behind", "standing beside", "standing in front of",
        "standing near", "standing next to", "staring at", "stuck in", "surrounding", "swimming in", "swinging",
        "talking to", "topped with", "touching", "traveling down", "traveling on", "tying", "typing on", "underneath",
        "wading in", "waiting for", "walking across", "walking by", "walking down", "walking next to",
        "walking through", "working in", "working on", "worn on", "wrapped around", "wrapped in", "by", "of", "near",
        "next to", "with", "beside", "on the side of", "around",
    }

    def __init__(self, transform: Transform, root_dir: str = AroVgDataset.DEFAULT_ROOT_DIR,
                 download: bool = True) -> None:
        super().__init__(transform=transform, annotation_filename="visual_genome_relation.json",
                         annotation_file_google_drive_id="1kX2iCHEv0CADL8dSO1nMdW-V0NqIAiP3", root_dir=root_dir,
                         download=download)

        for instance in self.instances:
            instance["class_name"] = instance["relation_name"]

        self.instances = [instance
                          for instance in self.instances
                          if instance["class_name"] not in self.SYMMETRIC_RELATIONSHIPS]


class AroVgAttribution(AroVgDataset):
    def __init__(self, transform: Transform, root_dir: str = AroVgDataset.DEFAULT_ROOT_DIR,
                 download: bool = True) -> None:
        super().__init__(transform=transform, annotation_filename="visual_genome_attribution.json",
                         annotation_file_google_drive_id="13tWvOrNOLHxl3Rm9cR3geAdHx2qR3-Tw", root_dir=root_dir,
                         download=download)

        for instance in self.instances:
            instance["class_name"] = f"{instance['attributes'][0]}_{instance['attributes'][1]}"


class AroOrderDataset(AroDataset, ABC):
    def __init__(self, root_dir: str, annotation_url: str, transform: Transform, download: bool = True,
                 max_words: int = 30, use_cache_for_spacy_docs: bool = True,
                 use_cache_for_instances: bool = True) -> None:
        if not os.path.exists(root_dir):
            if download:
                os.makedirs(os.path.dirname(root_dir), exist_ok=True)
                with FileLock(root_dir + ".lock"):
                    if not os.path.exists(root_dir):  # Retry in case the process had been locked.
                        logging.info(f"Downloading the data for the dataset {self.__class__.__name__}.")
                        self._download(root_dir)
            else:
                raise RuntimeError("Dataset not found.")

        cached_instances_path = os.path.join(root_dir, "instances.json")

        if use_cache_for_instances and os.path.exists(cached_instances_path):
            logging.info(f"Loading the cached instances from {cached_instances_path}…")
            with open(cached_instances_path) as file:
                instances = json.load(file)
        else:
            annotations_path = os.path.join(root_dir, os.path.basename(annotation_url))

            if not os.path.exists(annotations_path):
                with FileLock(annotations_path + ".lock"):
                    if not os.path.exists(annotations_path):  # Retry in case the process had been locked.
                        download_url(annotation_url, root_dir)

            with open(annotations_path) as file:
                instances_per_image = json.load(file)

            instances = [(instance["image"], caption)
                         for instance in instances_per_image
                         for caption in instance["caption"]]

            maybe_nltk_download("punkt", "tokenizers/punkt")

            perturb_functions = [(lambda text, spacy_doc: text), shuffle_nouns_and_adj, shuffle_all_but_nouns_and_adj,
                                 shuffle_within_trigrams, shuffle_trigrams]

            cached_spacy_docs_path = os.path.join(root_dir, "docs.spacy")

            if use_cache_for_spacy_docs and os.path.exists(cached_spacy_docs_path):
                logging.info(f"Loading the cached spaCy docs from {cached_spacy_docs_path}…")
                spacy_bin = spacy.tokens.DocBin().from_disk(cached_spacy_docs_path)
                assert len(spacy_bin) == len(instances)
                spacy_docs = spacy_bin.get_docs(NLP.vocab)
            else:
                spacy_docs = _get_spacy_docs(t[1] for t in instances)

                if use_cache_for_spacy_docs:
                    def _save_spacy_docs(spacy_docs: Iterator[spacy.tokens.Doc],
                                         path: str) -> Iterator[spacy.tokens.Doc]:
                        doc_bin = spacy.tokens.DocBin()
                        try:
                            for spacy_doc in spacy_docs:
                                doc_bin.add(spacy_doc)
                                yield spacy_doc
                        finally:
                            # The generator from the current function could be closed before reaching the end of the
                            # function.
                            # Recall the generators get closed when they are garbage collected,
                            # which typically occurs when their ref count goes to 0,
                            # generally when they get out of scope.
                            # For example, this can happen if not all the elements of the generator are consumed
                            # (which, in this case, would be a bug from the generator user) or, more realistically,
                            # if all the elements are consumed but then the generator was closed before the `for` loop
                            # reached the `StopIteration` point.
                            # The latter can occur because we somehow knew how many elements they were in the generator.
                            # For example, if we do a `zip` with an iterator and the current function (in that order),
                            # then this described scenario would occur,
                            # and no code after the last `yield` would be executed.
                            #
                            # In that case, `close()` was called and a `GeneratorExit` base-exception was raised, and
                            # then any remaining code wouldn't be executed.
                            # By putting the code in this `finally` block, we ensure it is executed unconditionally.
                            with FileLock(path + ".lock"):
                                if not os.path.exists(path):
                                    doc_bin.to_disk(path)
                                    logging.info(f"spaCy docs cache saved to {path}.")

                    spacy_docs = _save_spacy_docs(spacy_docs, cached_spacy_docs_path)

            instances = [
                {
                    "image_path": image_path,
                    "caption_options": [pre_caption(perturb_fn(caption, spacy_doc), max_words)
                                        for perturb_fn in perturb_functions],
                }
                for (image_path, caption), spacy_doc in zip(
                    tqdm(instances, desc=f"Processing the annotations from {self.__class__.__name__}"),
                    spacy_docs,
                )
            ]

            if use_cache_for_instances:
                with FileLock(cached_instances_path + ".lock"):
                    if not os.path.exists(cached_instances_path):
                        with open(cached_instances_path, "w") as file:
                            json.dump(instances, file)
                        logging.info(f"Cached instances saved to {cached_instances_path}.")

        super().__init__(instances=instances, root_dir=root_dir, transform=transform)

    @abstractmethod
    def _download(self, root_dir: str) -> None:
        raise NotImplementedError


class AroCocoOrder(AroOrderDataset):
    ANNOTATION_URLS = {
        "val": "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json",
        "test": "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json",
    }

    def __init__(self, transform: Transform,
                 root_dir: str = os.getenv("ARO_COCO_ORDER_ROOT_DIR", os.path.join(CACHE_DIR, "coco")),
                 split: Literal["val", "test"] = "test", download: bool = True, max_words: int = 30) -> None:
        super().__init__(root_dir=root_dir, annotation_url=self.ANNOTATION_URLS[split],
                         transform=transform, download=download, max_words=max_words)

    def _download(self, root_dir: str) -> None:
        os.makedirs(root_dir, exist_ok=True)
        subprocess.run(["wget", "http://images.cocodataset.org/zips/val2014.zip"], cwd=root_dir, check=True)
        subprocess.run(["unzip", "val2014.zip"], cwd=root_dir, check=True)
        # "test2014" is not used by "coco_karpathy_test.json".


class AroFlickr30kOrder(AroOrderDataset):
    ANNOTATION_URLS = {
        "val": "https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_val.json",
        "test": "https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_test.json",
    }

    def __init__(self, transform: Transform,
                 root_dir: str = os.getenv("ARO_FLICKR_ORDER_ROOT_DIR", os.path.join(CACHE_DIR, "flickr30k")),
                 split: Literal["val", "test"] = "test", download: bool = True, max_words: int = 30) -> None:
        super().__init__(root_dir=root_dir, annotation_url=self.ANNOTATION_URLS[split],
                         transform=transform, download=download, max_words=max_words)

    def _download(self, root_dir: str) -> None:
        s3_sync("[PATH]/flickr30k/images/",
                os.path.join(root_dir, "flickr30k-images").rstrip("/") + "/")
