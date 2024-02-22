import numbers
import random
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import PIL.Image
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import ImageFilter, ImageOps
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Normalize, RandomApply, RandomGrayscale, \
    RandomResizedCrop, Resize, ToTensor

from .constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from .utils import to_2tuple

Transform = Callable[[PIL.Image.Image], torch.Tensor]

Mode = Literal["RGB"]
Interpolation = Literal["bicubic", "bilinear", "random"]
ResizeMode = Literal["shortest", "longest", "squash"]


@dataclass
class PreprocessCfg:
    size: int | tuple[int, int] = 224
    mode: Mode = "RGB"
    mean: tuple[float, ...] = OPENAI_DATASET_MEAN
    std: tuple[float, ...] = OPENAI_DATASET_STD
    interpolation: Interpolation = "bicubic"
    resize_mode: ResizeMode = "shortest"
    fill_color: int = 0

    def __post_init__(self) -> None:
        assert self.mode in ("RGB",)

    @property
    def num_channels(self) -> int:
        return 3

    @property
    def input_size(self) -> tuple[int, int, int]:
        return (self.num_channels,) + to_2tuple(self.size)


_PREPROCESS_KEYS = set(asdict(PreprocessCfg()).keys())


def merge_preprocess_dict(base: PreprocessCfg | Mapping[str, Any],
                          overlay: Mapping[str, Any]) -> Mapping[str, Any]:
    """Merge overlay key-value pairs on top of the base preprocessing cfg or dict.
    Input dicts are filtered based on PreprocessCfg fields.
    """
    if isinstance(base, PreprocessCfg):
        base_clean = asdict(base)
    else:
        base_clean = {k: v for k, v in base.items() if k in _PREPROCESS_KEYS}
    if overlay:
        overlay_clean = {k: v for k, v in overlay.items() if k in _PREPROCESS_KEYS and v is not None}
        base_clean.update(overlay_clean)
    return base_clean


def merge_preprocess_kwargs(base: PreprocessCfg | Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
    return merge_preprocess_dict(base, kwargs)


@dataclass
class AugmentationCfg:
    scale: tuple[float, float] = (0.9, 1.0)
    ratio: tuple[float, float] | None = None
    color_jitter: float | tuple[float, float, float] | tuple[float, float, float, float] | None = None
    re_prob: float | None = None
    re_count: int | None = None
    use_timm: bool = False

    use_albumentations: bool = False
    num_crops: int = 1
    grayscale_prob: float | None = None
    blur_prob: float | None = None
    solarization_prob: float | None = None


def _setup_size(size: int | float | Sequence[int], error_msg: str) -> Sequence[int]:
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


class ResizeKeepRatio:
    """Resize and Keep Ratio

    Copy-pasted from `timm`.
    """

    def __init__(
            self,
            size: int | Sequence[int],
            longest: int | float = 0.,
            interpolation: InterpolationMode = InterpolationMode.BICUBIC,
            random_scale_prob: float = 0.,
            random_scale_range: tuple[float, float] = (0.85, 1.05),
            random_aspect_prob: float = 0.,
            random_aspect_range: tuple[float, float] = (0.9, 1.11),
    ) -> None:
        if isinstance(size, (list, tuple)):
            self.size = tuple(size)
        else:
            self.size = (size, size)
        self.interpolation = interpolation
        self.longest = float(longest)  # [0, 1] where 0 == shortest edge, 1 == longest
        self.random_scale_prob = random_scale_prob
        self.random_scale_range = random_scale_range
        self.random_aspect_prob = random_aspect_prob
        self.random_aspect_range = random_aspect_range

    @staticmethod
    def get_params(
            img: torch.Tensor | PIL.Image.Image,
            target_size: tuple[int, int],
            longest: float,
            random_scale_prob: float = 0.,
            random_scale_range: tuple[float, float] = (0.85, 1.05),
            random_aspect_prob: float = 0.,
            random_aspect_range: tuple[float, float] = (0.9, 1.11),
    ) -> Sequence[float]:
        """Get parameters
        """
        source_size = img.size[::-1]  # h, w
        h, w = source_size
        target_h, target_w = target_size
        ratio_h = h / target_h
        ratio_w = w / target_w
        ratio = max(ratio_h, ratio_w) * longest + min(ratio_h, ratio_w) * (1. - longest)
        if random_scale_prob > 0 and random.random() < random_scale_prob:
            ratio_factor = random.uniform(random_scale_range[0], random_scale_range[1])
            ratio_factor = (ratio_factor, ratio_factor)
        else:
            ratio_factor = (1., 1.)
        if random_aspect_prob > 0 and random.random() < random_aspect_prob:
            aspect_factor = random.uniform(random_aspect_range[0], random_aspect_range[1])
            ratio_factor = (ratio_factor[0] / aspect_factor, ratio_factor[1] * aspect_factor)
        size = [round(x * f / ratio) for x, f in zip(source_size, ratio_factor)]
        return size

    def __call__(self, img: PIL.Image.Image) -> PIL.Image.Image | torch.Tensor:
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Resized, padded to at least target size, possibly cropped to exactly target size
        """
        size = self.get_params(
            img, self.size, self.longest,
            self.random_scale_prob, self.random_scale_range,
            self.random_aspect_prob, self.random_aspect_range
        )
        img = F.resize(img, size, self.interpolation)  # noqa
        return img

    def __repr__(self) -> str:
        return self.__class__.__name__ + (f"(size={self.size}, interpolation={self.interpolation}),"
                                          f" longest={self.longest:.3f})")


def center_crop_or_pad(img: torch.Tensor, output_size: int | Sequence[int],
                       fill: int | tuple[int, ...] = 0) -> torch.Tensor:
    """Center crops and/or pads the given image.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.

    Args:
        img: Image to be cropped.
        output_size: (height, width) of the crop box.
            If int or sequence with single int, it is used for both directions.
        fill: Padding color

    Returns:
        PIL Image or Tensor: Cropped image.
    """
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    elif isinstance(output_size, (tuple, list)) and len(output_size) == 1:
        output_size = (output_size[0], output_size[0])

    _, image_height, image_width = F.get_dimensions(img)
    crop_height, crop_width = output_size

    if crop_width > image_width or crop_height > image_height:
        padding_ltrb = [
            (crop_width - image_width) // 2 if crop_width > image_width else 0,
            (crop_height - image_height) // 2 if crop_height > image_height else 0,
            (crop_width - image_width + 1) // 2 if crop_width > image_width else 0,
            (crop_height - image_height + 1) // 2 if crop_height > image_height else 0,
        ]
        img = F.pad(img, padding_ltrb, fill=fill)
        _, image_height, image_width = F.get_dimensions(img)
        if crop_width == image_width and crop_height == image_height:
            return img

    crop_top = int(round((image_height - crop_height) / 2.0))
    crop_left = int(round((image_width - crop_width) / 2.0))
    return F.crop(img, crop_top, crop_left, crop_height, crop_width)


class CenterCropOrPad(torch.nn.Module):
    """Crops the given image at the center.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
    """

    def __init__(self, size: Sequence[int] | int, fill: int | tuple[int, ...] = 0) -> None:
        super().__init__()
        self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")
        self.fill = fill

    def forward(self, img: PIL.Image.Image | torch.Tensor) -> PIL.Image.Image | torch.Tensor:
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        return center_crop_or_pad(img, self.size, fill=self.fill)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


class GaussianBlur:
    def __init__(self, p: float) -> None:
        self.p = p

    def __call__(self, img: PIL.Image.Image) -> PIL.Image.Image:
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization:
    def __call__(self, img: PIL.Image.Image) -> PIL.Image.Image:
        return ImageOps.solarize(img)


class MultiCrop:
    def __init__(self, transform: Callable[[Any], torch.Tensor], num_crops: int) -> None:
        self.transform = transform
        self.num_crops = num_crops

    def __call__(self, img: Any) -> torch.Tensor:
        return torch.stack([self.transform(img) for _ in range(self.num_crops)])


def _convert_to_rgb(image: PIL.Image.Image) -> PIL.Image.Image:
    return image.convert("RGB")


def _str_to_cv2_interpolation(interpolation: Literal["bicubic", "bilinear", "lanczos", "nearest"]) -> int:
    import cv2
    return {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
    }[interpolation]


def image_transform(
        image_size: int | tuple[int, int],
        is_train: bool,
        mean: tuple[float, ...] | None = None,
        std: tuple[float, ...] | None = None,
        resize_mode: ResizeMode = "shortest",
        interpolation: Interpolation = "bicubic",
        fill_color: int | tuple[int, ...] = 0,
        aug_cfg: Mapping[str, Any] | AugmentationCfg | None = None,
) -> Transform:
    mean = mean or OPENAI_DATASET_MEAN
    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3

    std = std or OPENAI_DATASET_STD
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3

    assert interpolation != "random"
    # NOTE random is ignored for interpolation_mode, so defaults to BICUBIC for inference if set
    interpolation_mode = InterpolationMode.BILINEAR if interpolation == "bilinear" else InterpolationMode.BICUBIC

    if isinstance(aug_cfg, dict):
        aug_cfg = AugmentationCfg(**aug_cfg)
    else:
        aug_cfg = aug_cfg or AugmentationCfg()

    normalize = Normalize(mean=mean, std=std)

    if is_train:
        aug_cfg_dict = {k: v for k, v in asdict(aug_cfg).items() if v is not None}

        num_crops = aug_cfg_dict.pop("num_crops", 1)

        use_timm = aug_cfg_dict.pop("use_timm", False)
        use_albumentations = aug_cfg_dict.pop("use_albumentations", False)
        if use_timm:
            assert not use_albumentations, "Can't use both timm and albumentations"

            from timm.data import create_transform  # timm can still be optional

            if isinstance(image_size, (tuple, list)):
                assert len(image_size) >= 2
                input_size = (3,) + image_size[-2:]
            else:
                input_size = (3, image_size, image_size)

            aug_cfg_dict.setdefault("color_jitter", None)  # disable by default

            # Pop out the extra transforms before creating the timm transforms.
            grayscale_prob = aug_cfg_dict.pop("grayscale_prob", 0)
            blur_prob = aug_cfg_dict.pop("blur_prob", 0)
            solarization_prob = aug_cfg_dict.pop("solarization_prob", 0)

            first_transforms, second_transforms, third_transforms = create_transform(
                input_size=input_size,
                is_training=True,
                hflip=0.,
                mean=mean,
                std=std,
                re_mode="pixel",
                interpolation=interpolation,
                **aug_cfg_dict,
            )

            extra_transforms = []

            if grayscale_prob:
                extra_transforms.append(RandomGrayscale(grayscale_prob))

            if blur_prob:
                extra_transforms.append(GaussianBlur(blur_prob))

            if solarization_prob:
                extra_transforms.append(RandomApply([Solarization()], p=solarization_prob))

            train_transform = Compose([first_transforms, second_transforms, *extra_transforms, third_transforms])
        elif use_albumentations:
            import albumentations as A
            import cv2
            from albumentations.pytorch import ToTensorV2

            if isinstance(image_size, (tuple, list)):
                assert len(image_size) >= 2
                height, width = image_size[-2:]
            else:
                height = width = image_size

            scale = aug_cfg_dict.pop("scale", (0.08, 1.0))
            ratio = aug_cfg_dict.pop("ratio", (3. / 4., 4. / 3.))
            interpolation = aug_cfg_dict.pop("interpolation", "random")

            if interpolation == "random":
                interpolation_transform = A.OneOf([
                    A.RandomResizedCrop(height=height, width=width, scale=scale, ratio=ratio,
                                        interpolation=cv2.INTER_LINEAR),
                    A.RandomResizedCrop(height=height, width=width, scale=scale, ratio=ratio,
                                        interpolation=cv2.INTER_CUBIC),
                ], p=1)
            else:
                cv2_interpolation = _str_to_cv2_interpolation(interpolation)
                interpolation_transform = A.RandomResizedCrop(height=height, width=width, scale=scale, ratio=ratio,
                                                              interpolation=cv2_interpolation),

            color_jitter = aug_cfg_dict.pop("color_jitter", (0, 0, 0, 0, 0))
            if isinstance(color_jitter, (list, tuple)):
                # color jitter should be a 3-tuple/list if spec brightness/contrast/saturation
                # or 4 if also augmenting hue
                # or 5 if specifying the probability
                assert len(color_jitter) in (3, 4, 5)
                if len(color_jitter) == 3:
                    color_jitter = color_jitter + (0, 1)
                elif len(color_jitter) == 4:
                    color_jitter = color_jitter + (1,)
                elif len(color_jitter) != 5:
                    raise ValueError(f"Invalid color_jitter length: {color_jitter}")
            else:
                # if it's a scalar, duplicate for brightness, contrast, and saturation, no hue, and prob 1
                color_jitter = (float(color_jitter),) * 3 + (0, 1)

            albumentations_transform = A.Compose([
                interpolation_transform,
                A.ColorJitter(*color_jitter[:4], p=color_jitter[-1]),
                A.ToGray(p=aug_cfg_dict.pop("grayscale_prob", 0)),
                A.GaussianBlur(blur_limit=0, sigma_limit=(0.1, 2), p=aug_cfg_dict.pop("blur_prob", 0)),
                A.Solarize(p=aug_cfg_dict.pop("solarization_prob", 0)),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ])

            train_transform = Compose([
                _convert_to_rgb,
                np.asarray,
                lambda img: albumentations_transform(image=img)["image"],
            ])
        else:
            train_transform = Compose([
                RandomResizedCrop(
                    image_size,
                    scale=aug_cfg_dict.pop("scale"),
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                _convert_to_rgb,
                ToTensor(),
                normalize,
            ])

        if aug_cfg_dict:
            warnings.warn(f"Unused augmentation cfg items: ({list(aug_cfg_dict.keys())}).")

        if num_crops == 1:
            return train_transform
        elif num_crops > 1:
            return MultiCrop(train_transform, num_crops)
        else:
            raise ValueError(f"Invalid num_crops: {num_crops}")
    else:
        if resize_mode == "longest":
            transforms = [
                ResizeKeepRatio(image_size, interpolation=interpolation_mode, longest=1),
                CenterCropOrPad(image_size, fill=fill_color)
            ]
        elif resize_mode == "squash":
            if isinstance(image_size, int):
                image_size = (image_size, image_size)
            transforms = [
                Resize(image_size, interpolation=interpolation_mode),
            ]
        elif resize_mode == "shortest":
            if not isinstance(image_size, (tuple, list)):
                image_size = (image_size, image_size)
            if image_size[0] == image_size[1]:
                # simple case, use torchvision built-in Resize w/ shortest-edge mode (scalar size arg)
                transforms = [
                    Resize(image_size[0], interpolation=interpolation_mode)
                ]
            else:
                # resize the shortest edge to matching target dim for non-square target
                transforms = [ResizeKeepRatio(image_size)]
            transforms += [CenterCrop(image_size)]
        else:
            raise ValueError(f"Invalid resize mode: {resize_mode}")

        transforms.extend([
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])
        return Compose(transforms)


def image_transform_v2(
        cfg: PreprocessCfg,
        is_train: bool,
        aug_cfg: Mapping[str, Any] | AugmentationCfg | None = None,
) -> Transform:
    return image_transform(
        image_size=cfg.size,
        is_train=is_train,
        mean=cfg.mean,
        std=cfg.std,
        interpolation=cfg.interpolation,
        resize_mode=cfg.resize_mode,
        fill_color=cfg.fill_color,
        aug_cfg=aug_cfg,
    )
