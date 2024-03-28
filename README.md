# CLoVe

This repo contains the code and pre-trained models that we released along with our paper:

[CLoVe: Encoding Compositional Language in Contrastive Vision-Language Models](https://arxiv.org/abs/2402.15021)

[Santiago Castro](https://santi.uy/),
[Amir Ziai](https://linkedin.com/in/amirziai),
[Avneesh Saluja](https://asaluja.github.io/),
[Zhuoning Yuan](https://zhuoning.cc/),
and [Rada Mihalcea](https://web.eecs.umich.edu/~mihalcea/).

**TL;DR:** CLoVe is a framework to significantly improve the ability of existing
[CLIP](https://openai.com/research/clip)-like models to encode compositional language while keeping or improving the 
performance on standard vision-language tasks.

This codebase is largely based on [OpenCLIP's](https://github.com/mlfoundations/open_clip),
containing its changes until (and including)
[the commit `73fa7f0`](https://github.com/mlfoundations/open_clip/commit/73fa7f0).
We are specially thankful to the authors of OpenCLIP for their work.

## Setup

It's recommended to have a CUDA-12-enabled GPU with the NVIDIA driver version 530 or greater with CUDA 12.1 or 
later installed.
If you don't have this, you need to change the [`pyproject.toml`](pyproject.toml) file to use a different version of
PyTorch.

With Python 3.10 or later, clone this repo and run:

```bash
pip install -e .
# export PYTHONPATH=src  # TODO: I think it's not necessary.
```

### HMDB51 and UCF101 Evaluation

You need to have [a rarfile backend](https://github.com/markokr/rarfile) installed (e.g., `unrar`).

For UCF101, given that [UCF's website server's certificate chain is
incomplete](https://www.ssllabs.com/ssltest/analyze.html?d=www.crcv.ucf.edu), you need to run the following
(note that this command runs two sudo commands to include an intermediate Certificate Authority certificate to the 
system):

```bash
./scripts/add_missing_ssl_certs.sh
```

### Winoground Evaluation

You need to be logged in to HuggingFace:

```bash
huggingface-cli login
```

You also need to [accept the terms of use for the dataset](https://huggingface.co/datasets/facebook/winoground).

### Run with Ray

If you want to run our code with [Ray](https://www.ray.io/), follow these steps.
First, you need to generate a `requirements.txt` file.
For it, you need to use [Poetry](https://python-poetry.org/) (which we use to define the high-level dependencies):

```bash
poetry self add poetry-plugin-export
poetry export --format requirements.txt --output requirements.txt --without-hashes
```

After this, see the example files under [`scripts/`](scripts), such as [`example_ray.sh`](scripts/example_ray.sh).

## Using the Pre-Trained Model

If you want to use the pre-trained model, do:

```python
import torch
from PIL import Image
from cached_path import cached_path

from open_clip import create_model_and_preprocessing
from training.file_utils import pt_load
from training.utils import get_state_dict, patch_model

model, _, transform, tokenizer = create_model_and_preprocessing("ViT-B-32", "openai")
model.eval()

URL = ("https://github.com/Netflix/clove/releases/download/pretrained/"
       "clove_without_patching.pt")
patch_model(model, get_state_dict(pt_load(URL), model), weight_for_state_dict=0.6)

image_path = cached_path(
    "https://github.com/mlfoundations/open_clip/blob/main/docs/CLIP.png?raw=true")
image = transform(Image.open(image_path)).unsqueeze(0)
text = tokenizer(["a diagram", "a dog", "a cat"])

with torch.inference_mode(), torch.cuda.amp.autocast():
    output = model(image, text)
    image_features, text_features = output["image_features"], output["text_features"]

    print("Label probs:", (100 * image_features @ text_features.T).softmax(dim=-1))
    # Prints `[[9.9900e-01, 7.4042e-04, 2.6385e-04]]`.
```

## Evaluating a Pre-Trained Model

To evaluate our model on all the benchmarks from the paper, run:

```bash
python -m training \
  --eval-benchmarks aro color didemo hmdb51 imagenet-v2 imagenet-val msrvtt sts sugar-crepe svo-probes ucf101 val \
  winoground youcook2 cb/wds/cars cb/wds/vtab/cifar10 cb/wds/vtab/cifar100 cb/wds/mnist \
  cb/wds/vtab/eurosat cb/wds/vtab/flowers cb/wds/vtab/dtd \
  --model ViT-B-32 \
  --pretrained openai \
  --wise-ft https://github.com/Netflix/clove/releases/download/pretrained/clove_without_patching.pt \
  --wise-ft-weight-for-2 0.6
```

You can list all the available program options by running:

```bash
python -m training --help
````

## Reproducing our Training Procedure

To reproduce our training (fine-tuning) procedure, you would need a machine with 8x GPUs with enough memory
(e.g., A10, A100, or A40; you may manage to reproduce similar results by adjusting some parameters)
and you need to follow these steps:

1. [Download LAION-COCO in the webdataset
    format](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/laion-coco.md).
2. Set its path in `_DATASET_SHORT_NAMES["laion-coco"]` in [`src/training/params.py`](src/training/params.py).
3. Run:

```bash
./scripts/example_multi_gpu.sh
```

If your machine has fewer than 8 GPUs or doesn't adapt well to the script, review it and change it accordingly.
See more training code examples under [`scripts/`](scripts).

You can list all the available program options by running:

```bash
python -m training --help
```

Also, see [OpenCLIP's repo](https://github.com/mlfoundations/open_clip) for more details on how to train models.

## Citation

```bibtex
@misc{clove,
  title={{CLoVe}: Encoding Compositional Language in Contrastive Vision-Language Models},
  author={Santiago Castro and Amir Ziai and Avneesh Saluja and Zhuoning Yuan and Rada Mihalcea},
  howpublished={arXiv:2402.15021},
  month=feb,
  year={2024},
  url={https://arxiv.org/abs/2402.15021},
  eprint={2402.15021},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

**If you use our code, please consider also
[citing OpenCLIP](https://github.com/mlfoundations/open_clip?tab=readme-ov-file#citing).**

---

## To do

* further test the setup and instructions
* add a figure at the beginning
* change the example code to be more compositionality-specific
* add a repo description and tags
* upload an already-patched model
  * maybe create a table with the available pre-trained weights and reference performance
* provide pre-trained weights for larger models
* make it easy to install as a library
* incorporate the weights in open_clip
