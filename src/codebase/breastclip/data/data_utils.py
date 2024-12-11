import os
from typing import Dict, Iterable
import numpy as np
import cv2
from PIL import Image

from albumentations import *
from torchvision.transforms import Compose as TorchCompose
from torchvision.transforms import Resize as TorchResize
from transformers import AutoTokenizer


def otsu_mask(img):
    median = np.median(img)
    _, thresh = cv2.threshold(img, median, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


class OtsuCut(object):

    def __init__(self, align_orientation: bool = False, remove_text: bool = False):
        super().__init__()
        self.algn_orientation = align_orientation
        self.remove_text = remove_text

    def __process__(self, x):
        if isinstance(x, Image.Image):
            x = np.array(x)
        
        mask = otsu_mask(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY))
        # Convert to NumPy array if not already

        # Check if the matrix is empty or has no '1's
        if mask.size == 0 or not np.any(mask):
            return Image.fromarray(x)

        # Find the rows and columns where '1' appears
        rows = np.any(mask == 255, axis=1)
        cols = np.any(mask == 255, axis=0)

        # Find the indices of the rows and columns
        min_row, max_row = np.where(rows)[0][[0, -1]]
        min_col, max_col = np.where(cols)[0][[0, -1]]

        # Crop and return the submatrix
        x = x[min_row:max_row+1, min_col:max_col+1]
        
        img = Image.fromarray(x)
        return img

    def __call__(self, image):
        if isinstance(x, Iterable):
            return [self.__process__(im) for im in x]
        else:
            return self.__process__(x)


def load_tokenizer(source, pretrained_model_name_or_path, cache_dir, **kwargs):
    if source == "huggingface":
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            cache_dir=cache_dir,
            local_files_only=os.path.exists(
                os.path.join(cache_dir, f'models--{pretrained_model_name_or_path.replace("/", "--")}')),
            **kwargs,
        )
        if tokenizer.bos_token_id is None:
            tokenizer.bos_token_id = tokenizer.cls_token_id
    else:
        raise KeyError(f"Not supported tokenizer source: {source}")

    return tokenizer


def load_transform(split: str = "train", transform_config: Dict = None):
    assert split in {"train", "valid", "test", "aug"}
    transforms = transform_config[split]
    if split == "train":
        if (transforms["Resize"]["size_h"] == 512 or transforms["Resize"]["size_h"] == 224) and (
                transforms["Resize"]["size_w"] == 512 or transforms["Resize"]["size_w"] == 224):
            return Compose([
                Resize(width=transforms["Resize"]["size_h"], height=transforms["Resize"]["size_w"]),
                HorizontalFlip(),
                VerticalFlip(),
                Affine(
                    rotate=transforms["transform"]["affine_transform_degree"],
                    translate_percent=transforms["transform"]["affine_translate_percent"],
                    scale=transforms["transform"]["affine_scale"],
                    shear=transforms["transform"]["affine_shear"]
                ),
                ElasticTransform(
                    alpha=transforms["transform"]["elastic_transform_alpha"],
                    sigma=transforms["transform"]["elastic_transform_sigma"]
                )
            ], p=transforms["transform"]["p"]
            )
        else:
            return Compose([
                HorizontalFlip(),
                VerticalFlip(),
                Affine(
                    rotate=transforms["transform"]["affine_transform_degree"],
                    translate_percent=transforms["transform"]["affine_translate_percent"],
                    scale=transforms["transform"]["affine_scale"],
                    shear=transforms["transform"]["affine_shear"]
                ),
                ElasticTransform(
                    alpha=transforms["transform"]["elastic_transform_alpha"],
                    sigma=transforms["transform"]["elastic_transform_sigma"]
                )
            ], p=transforms["transform"]["p"]
            )
    elif split == "valid":
        if transforms["Resize"]["size_h"] == 512 and transforms["Resize"]["size_w"] == 512:
            return Compose([
                Resize(width=transforms["Resize"]["size_h"], height=transforms["Resize"]["size_w"])
            ])
        else:
            return TorchCompose([
                OtsuCut(),
                TorchResize(size=(transforms["Resize"]["size_h"], transforms["Resize"]["size_w"]))
            ])

