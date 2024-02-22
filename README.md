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

This codebase is largely based on [OpenCLIP's](https://github.com/mlfoundations/open_clip) and is up-to-date until
[the commit `73fa7f0`](https://github.com/mlfoundations/open_clip/commit/73fa7f0).
We are specially thankful to the authors of OpenCLIP for their work.

## Setup

It's recommended to have a CUDA-12-enabled GPU with the NVIDIA driver version 530 or greater with CUDA 12.1 or 
later installed.
If you don't have this, you need to change the [`pyproject.toml`](pyproject.toml) file to use a different version of
PyTorch.

With Python 3.10 or later, clone this repo and run:

```bash
export PYTHONPATH=src
pip install -e .
```

### HMDB51 and UCF101 Evaluation

You need to have [a rarfile backend](https://github.com/markokr/rarfile) installed (e.g., `unrar`).

For UCF101, given that [UCF's website is that the server's certificate chain is
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

```bash
TODO
```

## Reproducing our Training Procedure

To reproduce our training (fine-tuning) procedure, you need to follow these steps:

```bash
TODO
```

See more examples of training code under [`scripts/`](scripts).

See all the available program options by running:

```bash
python -m training --help
````

## Evaluating a Pre-Trained Model

TODO

See all the available program options by running:

```bash
python -m training --help
````

## Citation

```bibtex
TODO
```

**If you use our code, please consider also
[citing OpenCLIP](https://github.com/mlfoundations/open_clip?tab=readme-ov-file#citing).**

---

TODO:

* arxiv link
* training and eval instructions
* citation
* upload the pretrained model
* replace `clove_pretrained_path`
* test setup and instructions
