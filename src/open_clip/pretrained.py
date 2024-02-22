import hashlib
import os
import urllib.request
import warnings
from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any

import filelock
import gdown
from tqdm.auto import tqdm

from .constants import IMAGENET_MEAN, IMAGENET_STD, INCEPTION_MEAN, INCEPTION_STD, OPENAI_DATASET_MEAN, \
    OPENAI_DATASET_STD
from .version import __version__

try:
    from huggingface_hub import hf_hub_download

    hf_hub_download = partial(hf_hub_download, library_name="open_clip", library_version=__version__)
    _has_hf_hub = True
except ImportError:
    hf_hub_download = None
    _has_hf_hub = False


CACHE_DIR = os.getenv("CACHE_DIR", "/mnt/.cache")


def _pcfg(url: str | None = None, hf_hub: str | None = None, **kwargs) -> Mapping[str, Any]:
    # OpenAI / OpenCLIP defaults
    return {
        "url": url,
        "hf_hub": hf_hub,
        "mean": OPENAI_DATASET_MEAN,
        "std": OPENAI_DATASET_STD,
        "interpolation": "bicubic",
        "resize_mode": "shortest",
        **kwargs,
    }


def _slpcfg(url: str | None = None, hf_hub: str | None = None, **kwargs) -> Mapping[str, Any]:
    # SigLIP defaults
    return {
        "url": url,
        "hf_hub": hf_hub,
        "mean": INCEPTION_MEAN,
        "std": INCEPTION_STD,
        "interpolation": "bicubic",
        "resize_mode": "squash",
        **kwargs,
    }


def _apcfg(url: str | None = None, hf_hub: str | None = None, **kwargs) -> Mapping[str, Any]:
    # CLIPA defaults
    return {
        "url": url,
        "hf_hub": hf_hub,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "interpolation": "bilinear",
        "resize_mode": "squash",
        **kwargs,
    }


_RN50 = dict(
    openai=_pcfg(
        "https://openaipublic.azureedge.net/clip/models/"
        "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt"),
    yfcc15m=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/rn50-quickgelu-yfcc15m-455df137.pt"),
    cc12m=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/rn50-quickgelu-cc12m-f000538c.pt"),
    cloob=_pcfg("https://ml.jku.at/research/CLOOB/downloads/checkpoints/cloob_rn50_yfcc_epoch_28.pt"),
    sugarcrepe_from_scratch_e30=_pcfg(gdrive_id="1BE9ngqexicC_G_FGMrOYOFX7RnVIgev6"),
    sugarcrepe_negate_from_scratch_e30=_pcfg(gdrive_id="1r4Iict3ir75FvZYCEYgJCGjBQ4bksMBl"),
    sugarcrepe_replace_from_scratch_e30=_pcfg(gdrive_id="1rM3VGmQOAmfQZbPBLZ_Opo4PdZZNzhSn"),
    sugarcrepe_swap_from_scratch_e30=_pcfg(gdrive_id="14TGP4sCWpP2KF3w31tzgfvRyow-IBV8V"),
)

_RN50_quickgelu = dict(
    openai=_pcfg(
        "https://openaipublic.azureedge.net/clip/models/"
        "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt"),
    yfcc15m=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/rn50-quickgelu-yfcc15m-455df137.pt"),
    cc12m=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/rn50-quickgelu-cc12m-f000538c.pt"),
)

_RN101 = dict(
    openai=_pcfg(
        "https://openaipublic.azureedge.net/clip/models/"
        "8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt"),
    yfcc15m=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/"
        "rn101-quickgelu-yfcc15m-3e04b30e.pt"),
)

_RN101_quickgelu = dict(
    openai=_pcfg(
        "https://openaipublic.azureedge.net/clip/models/"
        "8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt"),
    yfcc15m=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/"
        "rn101-quickgelu-yfcc15m-3e04b30e.pt"),
)

_RN50x4 = dict(
    openai=_pcfg(
        "https://openaipublic.azureedge.net/clip/models/"
        "7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt"),
)

_RN50x16 = dict(
    openai=_pcfg(
        "https://openaipublic.azureedge.net/clip/models/"
        "52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt"),
)

_RN50x64 = dict(
    openai=_pcfg(
        "https://openaipublic.azureedge.net/clip/models/"
        "be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt"),
)

_VITB32 = dict(
    openai=_pcfg(
        "https://openaipublic.azureedge.net/clip/models/"
        "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"),
    laion400m_e31=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/"
        "vit_b_32-quickgelu-laion400m_e31-d867053b.pt"),
    laion400m_e32=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/"
        "vit_b_32-quickgelu-laion400m_e32-46683a32.pt"),
    laion2b_e16=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-laion2b_e16-af8dbd0c.pth"),
    laion2b_s34b_b79k=_pcfg(hf_hub="laion/CLIP-ViT-B-32-laion2B-s34B-b79K/"),
    # DataComp-XL models
    datacomp_xl_s13b_b90k=_pcfg(hf_hub="laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K/"),
    # DataComp-M models
    datacomp_m_s128m_b4k=_pcfg(hf_hub="laion/CLIP-ViT-B-32-DataComp.M-s128M-b4K/"),
    commonpool_m_clip_s128m_b4k=_pcfg(hf_hub="laion/CLIP-ViT-B-32-CommonPool.M.clip-s128M-b4K/"),
    commonpool_m_laion_s128m_b4k=_pcfg(hf_hub="laion/CLIP-ViT-B-32-CommonPool.M.laion-s128M-b4K/"),
    commonpool_m_image_s128m_b4k=_pcfg(hf_hub="laion/CLIP-ViT-B-32-CommonPool.M.image-s128M-b4K/"),
    commonpool_m_text_s128m_b4k=_pcfg(hf_hub="laion/CLIP-ViT-B-32-CommonPool.M.text-s128M-b4K/"),
    commonpool_m_basic_s128m_b4k=_pcfg(hf_hub="laion/CLIP-ViT-B-32-CommonPool.M.basic-s128M-b4K/"),
    commonpool_m_s128m_b4k=_pcfg(hf_hub="laion/CLIP-ViT-B-32-CommonPool.M-s128M-b4K/"),
    # DataComp-S models
    datacomp_s_s13m_b4k=_pcfg(hf_hub="laion/CLIP-ViT-B-32-DataComp.S-s13M-b4K/"),
    commonpool_s_clip_s13m_b4k=_pcfg(hf_hub="laion/CLIP-ViT-B-32-CommonPool.S.clip-s13M-b4K/"),
    commonpool_s_laion_s13m_b4k=_pcfg(hf_hub="laion/CLIP-ViT-B-32-CommonPool.S.laion-s13M-b4K/"),
    commonpool_s_image_s13m_b4k=_pcfg(hf_hub="laion/CLIP-ViT-B-32-CommonPool.S.image-s13M-b4K/"),
    commonpool_s_text_s13m_b4k=_pcfg(hf_hub="laion/CLIP-ViT-B-32-CommonPool.S.text-s13M-b4K/"),
    commonpool_s_basic_s13m_b4k=_pcfg(hf_hub="laion/CLIP-ViT-B-32-CommonPool.S.basic-s13M-b4K/"),
    commonpool_s_s13m_b4k=_pcfg(hf_hub="laion/CLIP-ViT-B-32-CommonPool.S-s13M-b4K/"),

    negclip=_pcfg(gdrive_id="1ooVVPxB-tvptgmHlIMMFGV3Cg-IrhbRZ"),

    sugarcrepe_finetuned_e1=_pcfg(gdrive_id="1QRExPbF7YZI5Av13ul5i4b_IsNiqQOMI"),
    sugarcrepe_finetuned_e2=_pcfg(gdrive_id="16hOTvxIq4nMbJe_SsL4vtKdWdKlTf5x4"),
    sugarcrepe_finetuned_e3=_pcfg(gdrive_id="1mo-IX-wKPf8w0BtU8cvsVFSLfYpaIS-r"),
    sugarcrepe_finetuned_e4=_pcfg(gdrive_id="11xrFUKOOHrBwM5r6WfoDpY869J1KCOo4"),
    sugarcrepe_finetuned_e5=_pcfg(gdrive_id="1FAdRNrpfXby2mNfqx2Q-FZjtBm7NTSzS"),
    sugarcrepe_negate_finetuned_e1=_pcfg(gdrive_id="1sNw0uyDeF9tRwnTsi4W5xSRezCf0SgLW"),
    sugarcrepe_negate_finetuned_e2=_pcfg(gdrive_id="1_oLE7F-tl0XD7QbTu6EyPhgeNZmdUecX"),
    sugarcrepe_negate_finetuned_e3=_pcfg(gdrive_id="1_ALHkhls7y4_Ycmb5eWHv1fdL4ZWYREV"),
    sugarcrepe_negate_finetuned_e4=_pcfg(gdrive_id="1DcTtkaLIuxtY6Kp3NJ00G99svbKPg7M0"),
    sugarcrepe_negate_finetuned_e5=_pcfg(gdrive_id="1RDwxQctUsOI1-n88k8ok5Vq7KDeY_Kdp"),
    sugarcrepe_replace_finetuned_e1=_pcfg(gdrive_id="1VttyDEKK1rUD26Hg3MUOd7tM9F6OOqZ5"),
    sugarcrepe_replace_finetuned_e2=_pcfg(gdrive_id="1wbE_vT429Egh1VuciEpmeAyOq-e_NRje"),
    sugarcrepe_replace_finetuned_e3=_pcfg(gdrive_id="1i-pquy6PfBQlIKe162x7sIXYF_s01pcS"),
    sugarcrepe_replace_finetuned_e4=_pcfg(gdrive_id="19Z9bNoAapD-X66eStC4ZluB35QE8n5ce"),
    sugarcrepe_replace_finetuned_e5=_pcfg(gdrive_id="13XzRYAtvQAYRcVd2ki0r31lvz3fh0IQN"),
    sugarcrepe_swap_finetuned_e1=_pcfg(gdrive_id="1AKy4cJPubPYEeXU-PZ7yY9UaeStOmP0Y"),
    sugarcrepe_swap_finetuned_e2=_pcfg(gdrive_id="1LJwOvDBPMF9FZ_l7vU3t74GhlZ6LHxZK"),
    sugarcrepe_swap_finetuned_e3=_pcfg(gdrive_id="1d2mzsZ8rsQOmFU5_WxksRpk3-PAHtXVF"),
    sugarcrepe_swap_finetuned_e4=_pcfg(gdrive_id="1gDlLV7hwve8BsWWphlJqcZgjD8hcOAvQ"),
    sugarcrepe_swap_finetuned_e5=_pcfg(gdrive_id="1IaOf0q9rBH5_PEJexkaTclvLoRvW32Oh"),
)

_VITB32_quickgelu = dict(
    openai=_pcfg(
        "https://openaipublic.azureedge.net/clip/models/"
        "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"),
    laion400m_e31=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/"
        "vit_b_32-quickgelu-laion400m_e31-d867053b.pt"),
    laion400m_e32=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/"
        "vit_b_32-quickgelu-laion400m_e32-46683a32.pt"),
    metaclip_400m=_pcfg("https://dl.fbaipublicfiles.com/MMPT/metaclip/b32_400m.pt"),
    metaclip_fullcc=_pcfg("https://dl.fbaipublicfiles.com/MMPT/metaclip/b32_fullcc2.5b.pt"),
)

_VITB32_256 = dict(
    datacomp_s34b_b86k=_pcfg(hf_hub="laion/CLIP-ViT-B-32-256x256-DataComp-s34B-b86K/"),
)

_VITB16 = dict(
    openai=_pcfg(
        "https://openaipublic.azureedge.net/clip/models/"
        "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt"),
    laion400m_e31=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_16-laion400m_e31-00efa78f.pt"),
    laion400m_e32=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_16-laion400m_e32-55e67d44.pt"),
    laion2b_s34b_b88k=_pcfg(hf_hub="laion/CLIP-ViT-B-16-laion2B-s34B-b88K/"),
    # DataComp-XL models
    datacomp_xl_s13b_b90k=_pcfg(hf_hub="laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K/"),
    # DataComp-L models
    datacomp_l_s1b_b8k=_pcfg(hf_hub="laion/CLIP-ViT-B-16-DataComp.L-s1B-b8K/"),
    commonpool_l_clip_s1b_b8k=_pcfg(hf_hub="laion/CLIP-ViT-B-16-CommonPool.L.clip-s1B-b8K/"),
    commonpool_l_laion_s1b_b8k=_pcfg(hf_hub="laion/CLIP-ViT-B-16-CommonPool.L.laion-s1B-b8K/"),
    commonpool_l_image_s1b_b8k=_pcfg(hf_hub="laion/CLIP-ViT-B-16-CommonPool.L.image-s1B-b8K/"),
    commonpool_l_text_s1b_b8k=_pcfg(hf_hub="laion/CLIP-ViT-B-16-CommonPool.L.text-s1B-b8K/"),
    commonpool_l_basic_s1b_b8k=_pcfg(hf_hub="laion/CLIP-ViT-B-16-CommonPool.L.basic-s1B-b8K/"),
    commonpool_l_s1b_b8k=_pcfg(hf_hub="laion/CLIP-ViT-B-16-CommonPool.L-s1B-b8K/"),
    # DFN
    dfn2b=_pcfg(hf_hub="apple/DFN2B-CLIP-ViT-B-16/")
)

_VITB16_quickgelu = dict(
    openai=_pcfg(
        "https://openaipublic.azureedge.net/clip/models/"
        "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt"),
    metaclip_400m=_pcfg("https://dl.fbaipublicfiles.com/MMPT/metaclip/b16_400m.pt"),
    metaclip_fullcc=_pcfg("https://dl.fbaipublicfiles.com/MMPT/metaclip/b16_fullcc2.5b.pt"),
)

_VITB16_PLUS_240 = dict(
    laion400m_e31=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/"
        "vit_b_16_plus_240-laion400m_e31-8fb26589.pt"),
    laion400m_e32=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/"
        "vit_b_16_plus_240-laion400m_e32-699c4b84.pt"),
)

_VITL14 = dict(
    openai=_pcfg(
        "https://openaipublic.azureedge.net/clip/models/"
        "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt"),
    laion400m_e31=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_l_14-laion400m_e31-69988bb6.pt"),
    laion400m_e32=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_l_14-laion400m_e32-3d133497.pt"),
    laion2b_s32b_b82k=_pcfg(hf_hub="laion/CLIP-ViT-L-14-laion2B-s32B-b82K/", mean=INCEPTION_MEAN, std=INCEPTION_STD),
    # DataComp-XL models
    datacomp_xl_s13b_b90k=_pcfg(hf_hub="laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K/"),
    commonpool_xl_clip_s13b_b90k=_pcfg(hf_hub="laion/CLIP-ViT-L-14-CommonPool.XL.clip-s13B-b90K/"),
    commonpool_xl_laion_s13b_b90k=_pcfg(hf_hub="laion/CLIP-ViT-L-14-CommonPool.XL.laion-s13B-b90K/"),
    commonpool_xl_s13b_b90k=_pcfg(hf_hub="laion/CLIP-ViT-L-14-CommonPool.XL-s13B-b90K/"),
)

_VITL14_quickgelu = dict(
    openai=_pcfg(
        "https://openaipublic.azureedge.net/clip/models/"
        "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt"),
    metaclip_400m=_pcfg("https://dl.fbaipublicfiles.com/MMPT/metaclip/l14_400m.pt"),
    metaclip_fullcc=_pcfg("https://dl.fbaipublicfiles.com/MMPT/metaclip/l14_fullcc2.5b.pt"),
    dfn2b=_pcfg(hf_hub="apple/DFN2B-CLIP-ViT-L-14/"),
)

_VITL14_336 = dict(
    openai=_pcfg(
        "https://openaipublic.azureedge.net/clip/models/"
        "3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt"),
)

_VITL14_336_quickgelu = dict(
    openai=_pcfg(
        "https://openaipublic.azureedge.net/clip/models/"
        "3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt"),
)

_VITH14 = dict(
    laion2b_s32b_b79k=_pcfg(hf_hub="laion/CLIP-ViT-H-14-laion2B-s32B-b79K/"),
)

_VITH14_quickgelu = dict(
    metaclip_fullcc=_pcfg("https://dl.fbaipublicfiles.com/MMPT/metaclip/h14_fullcc2.5b.pt"),
    dfn5b=_pcfg(
        hf_hub="apple/DFN5B-CLIP-ViT-H-14/",
        interpolation="bicubic",
        resize_mode="squash"
    ),
)

_VITH14_378_quickgelu = dict(
    dfn5b=_pcfg(
        hf_hub="apple/DFN5B-CLIP-ViT-H-14-378/",
        interpolation="bicubic",
        resize_mode="squash"
    ),
)

_VITg14 = dict(
    laion2b_s12b_b42k=_pcfg(hf_hub="laion/CLIP-ViT-g-14-laion2B-s12B-b42K/"),
    laion2b_s34b_b88k=_pcfg(hf_hub="laion/CLIP-ViT-g-14-laion2B-s34B-b88K/"),
)

_VITbigG14 = dict(
    laion2b_s39b_b160k=_pcfg(hf_hub="laion/CLIP-ViT-bigG-14-laion2B-39B-b160k/"),
)

_robertaViTB32 = dict(
    laion2b_s12b_b32k=_pcfg(hf_hub="laion/CLIP-ViT-B-32-roberta-base-laion2B-s12B-b32k/"),
)

_xlmRobertaBaseViTB32 = dict(
    laion5b_s13b_b90k=_pcfg(hf_hub="laion/CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k/"),
)

_xlmRobertaLargeFrozenViTH14 = dict(
    frozen_laion5b_s13b_b90k=_pcfg(hf_hub="laion/CLIP-ViT-H-14-frozen-xlm-roberta-large-laion5B-s13B-b90k/"),
)

_convnext_base = dict(
    laion400m_s13b_b51k=_pcfg(hf_hub="laion/CLIP-convnext_base-laion400M-s13B-b51K/"),
)

_convnext_base_w = dict(
    laion2b_s13b_b82k=_pcfg(hf_hub="laion/CLIP-convnext_base_w-laion2B-s13B-b82K/"),
    laion2b_s13b_b82k_augreg=_pcfg(hf_hub="laion/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg/"),
    laion_aesthetic_s13b_b82k=_pcfg(hf_hub="laion/CLIP-convnext_base_w-laion_aesthetic-s13B-b82K/"),
)

_convnext_base_w_320 = dict(
    laion_aesthetic_s13b_b82k=_pcfg(hf_hub="laion/CLIP-convnext_base_w_320-laion_aesthetic-s13B-b82K/"),
    laion_aesthetic_s13b_b82k_augreg=_pcfg(hf_hub="laion/CLIP-convnext_base_w_320-laion_aesthetic-s13B-b82K-augreg/"),
)

_convnext_large_d = dict(
    laion2b_s26b_b102k_augreg=_pcfg(hf_hub="laion/CLIP-convnext_large_d.laion2B-s26B-b102K-augreg/"),
)

_convnext_large_d_320 = dict(
    laion2b_s29b_b131k_ft=_pcfg(hf_hub="laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft/"),
    laion2b_s29b_b131k_ft_soup=_pcfg(hf_hub="laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup/"),
)

_convnext_xxlarge = dict(
    laion2b_s34b_b82k_augreg=_pcfg(hf_hub="laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg/"),
    laion2b_s34b_b82k_augreg_rewind=_pcfg(hf_hub="laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-rewind/"),
    laion2b_s34b_b82k_augreg_soup=_pcfg(hf_hub="laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup/"),
)

_coca_VITB32 = dict(
    laion2b_s13b_b90k=_pcfg(hf_hub="laion/CoCa-ViT-B-32-laion2B-s13B-b90k/"),
    mscoco_finetuned_laion2b_s13b_b90k=_pcfg(hf_hub="laion/mscoco_finetuned_CoCa-ViT-B-32-laion2B-s13B-b90k/")
)

_coca_VITL14 = dict(
    laion2b_s13b_b90k=_pcfg(hf_hub="laion/CoCa-ViT-L-14-laion2B-s13B-b90k/"),
    mscoco_finetuned_laion2b_s13b_b90k=_pcfg(hf_hub="laion/mscoco_finetuned_CoCa-ViT-L-14-laion2B-s13B-b90k/")
)

_PRETRAINED = {
    "RN50": _RN50,
    "RN50-quickgelu": _RN50_quickgelu,
    "RN101": _RN101,
    "RN101-quickgelu": _RN101_quickgelu,
    "RN50x4": _RN50x4,
    "RN50x16": _RN50x16,
    "RN50x64": _RN50x64,

    "ViT-B-32": _VITB32,
    "ViT-B-32-256": _VITB32_256,
    "ViT-B-32-quickgelu": _VITB32_quickgelu,

    "ViT-B-16": _VITB16,
    "ViT-B-16-quickgelu": _VITB16_quickgelu,
    "ViT-B-16-plus-240": _VITB16_PLUS_240,
    "ViT-L-14": _VITL14,
    "ViT-L-14-quickgelu": _VITL14_quickgelu,
    "ViT-L-14-336": _VITL14_336,
    "ViT-L-14-336-quickgelu": _VITL14_336_quickgelu,
    "ViT-H-14": _VITH14,
    "ViT-H-14-quickgelu": _VITH14_quickgelu,
    "ViT-H-14-378-quickgelu": _VITH14_378_quickgelu,
    "ViT-g-14": _VITg14,
    "ViT-bigG-14": _VITbigG14,

    "roberta-ViT-B-32": _robertaViTB32,
    "xlm-roberta-base-ViT-B-32": _xlmRobertaBaseViTB32,
    "xlm-roberta-large-ViT-H-14": _xlmRobertaLargeFrozenViTH14,

    "convnext_base": _convnext_base,
    "convnext_base_w": _convnext_base_w,
    "convnext_base_w_320": _convnext_base_w_320,
    "convnext_large_d": _convnext_large_d,
    "convnext_large_d_320": _convnext_large_d_320,
    "convnext_xxlarge": _convnext_xxlarge,

    "coca_ViT-B-32": _coca_VITB32,
    "coca_ViT-L-14": _coca_VITL14,

    "EVA01-g-14": dict(
        # from QuanSun/EVA-CLIP/EVA01_CLIP_g_14_psz14_s11B.pt
        laion400m_s11b_b41k=_pcfg(hf_hub="timm/eva_giant_patch14_clip_224.laion400m_s11b_b41k/"),
    ),
    "EVA01-g-14-plus": dict(
        # from QuanSun/EVA-CLIP/EVA01_CLIP_g_14_plus_psz14_s11B.pt
        merged2b_s11b_b114k=_pcfg(hf_hub="timm/eva_giant_patch14_plus_clip_224.merged2b_s11b_b114k/"),
    ),
    "EVA02-B-16": dict(
        # from QuanSun/EVA-CLIP/EVA02_CLIP_B_psz16_s8B.pt
        merged2b_s8b_b131k=_pcfg(hf_hub="timm/eva02_base_patch16_clip_224.merged2b_s8b_b131k/"),
    ),
    "EVA02-L-14": dict(
        # from QuanSun/EVA-CLIP/EVA02_CLIP_L_psz14_s4B.pt
        merged2b_s4b_b131k=_pcfg(hf_hub="timm/eva02_large_patch14_clip_224.merged2b_s4b_b131k/"),
    ),
    "EVA02-L-14-336": dict(
        # from QuanSun/EVA-CLIP/EVA02_CLIP_L_336_psz14_s6B.pt
        merged2b_s6b_b61k=_pcfg(hf_hub="timm/eva02_large_patch14_clip_336.merged2b_s6b_b61k/"),
    ),
    "EVA02-E-14": dict(
        # from QuanSun/EVA-CLIP/EVA02_CLIP_E_psz14_s4B.pt
        laion2b_s4b_b115k=_pcfg(hf_hub="timm/eva02_enormous_patch14_clip_224.laion2b_s4b_b115k/"),
    ),
    "EVA02-E-14-plus": dict(
        # from QuanSun/EVA-CLIP/EVA02_CLIP_E_psz14_plus_s9B.pt
        laion2b_s9b_b144k=_pcfg(hf_hub="timm/eva02_enormous_patch14_plus_clip_224.laion2b_s9b_b144k/"),
    ),

    "ViT-B-16-SigLIP": dict(webli=_slpcfg(hf_hub="timm/ViT-B-16-SigLIP/")),
    "ViT-B-16-SigLIP-256": dict(webli=_slpcfg(hf_hub="timm/ViT-B-16-SigLIP-256/")),
    "ViT-B-16-SigLIP-i18n-256": dict(webli=_slpcfg(hf_hub="timm/ViT-B-16-SigLIP-i18n-256/")),
    "ViT-B-16-SigLIP-384": dict(webli=_slpcfg(hf_hub="timm/ViT-B-16-SigLIP-384/")),
    "ViT-B-16-SigLIP-512": dict(webli=_slpcfg(hf_hub="timm/ViT-B-16-SigLIP-512/")),
    "ViT-L-16-SigLIP-256": dict(webli=_slpcfg(hf_hub="timm/ViT-L-16-SigLIP-256/")),
    "ViT-L-16-SigLIP-384": dict(webli=_slpcfg(hf_hub="timm/ViT-L-16-SigLIP-384/")),
    "ViT-SO400M-14-SigLIP": dict(webli=_slpcfg(hf_hub="timm/ViT-SO400M-14-SigLIP/")),
    "ViT-SO400M-14-SigLIP-384": dict(webli=_slpcfg(hf_hub="timm/ViT-SO400M-14-SigLIP-384/")),

    "ViT-L-14-CLIPA": dict(datacomp1b=_apcfg(hf_hub="UCSC-VLAA/ViT-L-14-CLIPA-datacomp1B/")),
    "ViT-L-14-CLIPA-336": dict(datacomp1b=_apcfg(hf_hub="UCSC-VLAA/ViT-L-14-CLIPA-336-datacomp1B/")),
    "ViT-H-14-CLIPA": dict(datacomp1b=_apcfg(hf_hub="UCSC-VLAA/ViT-H-14-CLIPA-datacomp1B/")),
    "ViT-H-14-CLIPA-336": dict(
        laion2b=_apcfg(hf_hub="UCSC-VLAA/ViT-H-14-CLIPA-336-laion2B/"),
        datacomp1b=_apcfg(hf_hub="UCSC-VLAA/ViT-H-14-CLIPA-336-datacomp1B/"),
    ),
    "ViT-bigG-14-CLIPA": dict(datacomp1b=_apcfg(hf_hub="UCSC-VLAA/ViT-bigG-14-CLIPA-datacomp1B/")),
    "ViT-bigG-14-CLIPA-336": dict(datacomp1b=_apcfg(hf_hub="UCSC-VLAA/ViT-bigG-14-CLIPA-336-datacomp1B/")),

    "nllb-clip-base": dict(v1=_pcfg(hf_hub="visheratin/nllb-clip-base-oc/")),
    "nllb-clip-large": dict(v1=_pcfg(hf_hub="visheratin/nllb-clip-large-oc/")),

    "nllb-clip-base-siglip": dict(v1=_slpcfg(hf_hub='visheratin/nllb-clip-base-siglip/')),
    "nllb-clip-large-siglip": dict(v1=_slpcfg(hf_hub='visheratin/nllb-clip-large-siglip/')),
}


def _clean_tag(tag: str) -> str:
    # normalize pretrained tags
    return tag.lower().replace("-", "_")


def list_pretrained(as_str: bool = False) -> Sequence[str]:
    """Returns a list of pretrained models.

    Returns a tuple (model_name, pretrain_tag) by default or "name:tag" if `as_str == True`.
    """
    return [f"{k}:{t}" if as_str else (k, t) for k in _PRETRAINED.keys() for t in _PRETRAINED[k].keys()]


def list_pretrained_models_by_tag(tag: str) -> Sequence[str]:
    """Return all models having the specified pretrained tag."""
    tag = _clean_tag(tag)
    return [k for k, v in _PRETRAINED.items() if tag in v]


def list_pretrained_tags_by_model(model: str) -> Sequence[str]:
    """Return all pretrained tags for the specified model architecture."""
    return list(_PRETRAINED.get(model, {}).keys())


def is_pretrained_cfg(model: str, tag: str) -> bool:
    return _clean_tag(tag) in _PRETRAINED.get(model, {})


def get_pretrained_cfg(model: str, tag: str) -> Mapping[str, Any]:
    return _PRETRAINED.get(model, {}).get(_clean_tag(tag), {})


def get_pretrained_url(model: str, tag: str) -> str | None:
    return get_pretrained_cfg(model, _clean_tag(tag)).get("url")


BUFFER_SIZE = 8192


def download_pretrained_from_url(
        url: str,
        cache_dir: str | None = None,
) -> str:
    cache_dir = cache_dir or os.path.join(CACHE_DIR, "open_clip")
    os.makedirs(cache_dir, exist_ok=True)
    filename = os.path.basename(url)

    if "openaipublic" in url:
        expected_sha256 = url.split("/")[-2]
    elif "mlfoundations" in url:
        expected_sha256 = os.path.splitext(filename)[0].split("-")[-1]
    else:
        expected_sha256 = ""

    download_target = os.path.join(cache_dir, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    computed_hash = hashlib.sha256() if expected_sha256 else None

    if os.path.isfile(download_target):
        if expected_sha256:
            with open(download_target, "rb") as file:
                while buffer := file.read(BUFFER_SIZE):
                    computed_hash.update(buffer)

            if computed_hash.hexdigest().startswith(expected_sha256):
                return download_target
            else:
                warnings.warn(
                    f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")
        else:
            return download_target

    with filelock.FileLock(download_target + ".lock"):
        if not os.path.exists(download_target):
            with urllib.request.urlopen(url) as source, open(download_target, "wb") as output, \
                    tqdm(total=int(source.headers.get("Content-Length")), unit="iB", unit_scale=True) as p_bar:
                while buffer := source.read(BUFFER_SIZE):
                    output.write(buffer)

                    if computed_hash:
                        computed_hash.update(buffer)

                    p_bar.update(len(buffer))

    if expected_sha256 and not computed_hash.hexdigest().startswith(expected_sha256):
        raise RuntimeError("The model has been downloaded but the SHA256 checksum does not not match")

    return download_target


def has_hf_hub(necessary: bool = False) -> bool:
    if not _has_hf_hub and necessary:
        # if no HF Hub module installed, and it is necessary to continue, raise error
        raise RuntimeError(
            "Hugging Face hub model specified but package not installed. Run `pip install huggingface_hub`.")
    return _has_hf_hub


def download_pretrained_from_hf(
        model_id: str,
        filename: str = "open_clip_pytorch_model.bin",
        revision: str | None = None,
        cache_dir: str | None = None,
) -> str:
    has_hf_hub(True)
    return hf_hub_download(model_id, filename, revision=revision, cache_dir=cache_dir)


def download_pretrained_from_gdrive(gdrive_id: str, cache_dir: str | None = None) -> str:
    cache_dir = cache_dir or os.path.join(CACHE_DIR, "open_clip", "gdrive")
    os.makedirs(cache_dir, exist_ok=True)

    path = os.path.join(cache_dir, gdrive_id)

    if not os.path.exists(path):
        with filelock.FileLock(path + ".lock"):
            if not os.path.exists(path):
                gdown.download(id=gdrive_id, output=path, resume=True, use_cookies=False)

    return path


def download_pretrained(cfg: Mapping[str, Any], force_hf_hub: bool = False,
                        cache_dir: str | None = None) -> str | None:
    download_url = cfg.get("url")
    download_hf_hub = cfg.get("hf_hub")
    download_gdrive_id = cfg.get("gdrive_id")

    if download_hf_hub and force_hf_hub:
        download_url = None  # Use the HF hub even if the URL exists.

    if download_url:
        return download_pretrained_from_url(download_url, cache_dir=cache_dir)
    elif download_hf_hub:
        has_hf_hub(True)
        # We assume the hf_hub entries in pretrained config combine model_id + filename in
        # "org/model_name/filename.pt" form. To specify just the model id w/o filename and
        # use "open_clip_pytorch_model.bin" default, there must be a trailing slash "org/model_name/".
        model_id, filename = os.path.split(download_hf_hub)

        kwargs = {}
        if filename:
            kwargs["filename"] = filename
        return download_pretrained_from_hf(model_id, cache_dir=cache_dir, **kwargs)
    elif download_gdrive_id:
        return download_pretrained_from_gdrive(download_gdrive_id, cache_dir=cache_dir)
    else:
        return None
