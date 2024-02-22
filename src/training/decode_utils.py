"""Utilities for decoding batches of images and texts.

To use it, you can place a breakpoint in the training loop, then run:

```python
from .decode_utils import decode_batch, decode_image, decode_images, decode_texts
```

Then run the following as a variable to watch:

```python
decode_batch(*batch, tokenizer)
```

If you have PyCharm's plugin OpenCV Image Viewer installed, you can select the NumPy arrays from the result and press
`Alt+I` to visualize them as images.
"""
from collections.abc import Iterable, Sequence

import numpy as np
import torch

from open_clip import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from open_clip.tokenizer import HFTokenizer


def decode_texts(texts: torch.Tensor, tokenizer: HFTokenizer) -> Sequence[str]:
    return tokenizer.batch_decode(texts, skip_special_tokens=True)


def decode_image(image: torch.Tensor) -> np.ndarray:
    return ((image.numpy(force=True).transpose(1, 2, 0)
             * np.array(OPENAI_DATASET_STD))
            + np.array(OPENAI_DATASET_MEAN))[..., ::-1]


def decode_images(images: Iterable[torch.Tensor]) -> Sequence[np.ndarray]:
    return [decode_image(image) for image in images]


def decode_batch(images: torch.Tensor, texts: torch.Tensor,
                 tokenizer: HFTokenizer) -> Sequence[tuple[np.ndarray, str] | str]:
    return list(zip(decode_images(images), decode_texts(texts, tokenizer))) + list(decode_texts(texts[len(images):],
                                                                                                tokenizer))
