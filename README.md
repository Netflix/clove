# CLoVe

This repo contains the code and pre-trained models that accompanies our paper:

[CLoVe: Encoding Compositional Language in Contrastive Vision-Language Models](https://arxiv.org/abs/TODO)

[Santiago Castro](https://santi.uy/),
[Amir Ziai](https://linkedin.com/in/amirziai),
[Avneesh Saluja](https://asaluja.github.io/),
[Zhuoning Yuan](https://zhuoning.cc/),
and [Rada Mihalcea](https://web.eecs.umich.edu/~mihalcea/).

**TL;DR:** CLoVe is a framework to significantly improve the ability of existing
[CLIP](https://openai.com/research/clip)-like models to encode compositional language while keeping or improving the 
performance on standard vision-language tasks.

TODO: some example figure showing what our method enables?

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
# export PYTHONPATH=src  # I think it's not necessary.
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
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
tokenizer = open_clip.get_tokenizer("ViT-B-32")

model.eval()

# TODO: do patching.

image = preprocess(Image.open("CLIP.png")).unsqueeze(0)
text = tokenizer(["a diagram", "a dog", "a cat"])

with torch.inference_mode(), torch.cuda.amp.autocast():
    output = model(image)
    image_features, text_features = output["image_features"], output["text_features"]

    print("Label probs:", (image_features @ text_features.T).softmax(dim=-1))  # Prints `[[1., 0., 0.]]`.
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
  --wise-ft "$CLOVE_PATH" \
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
````

Also, see [OpenCLIP's repo](https://github.com/mlfoundations/open_clip) for more details on how to train models.

## Citation

```bibtex
TODO
```

**If you use our code, please consider also
[citing OpenCLIP](https://github.com/mlfoundations/open_clip?tab=readme-ov-file#citing).**

---

TODO:

* arxiv link
* how to use the pre-trained models
* eval instructions
* citation
* upload the pretrained model
* replace `clove_pretrained_path`
* further test the setup and instructions
