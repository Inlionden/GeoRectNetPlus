"""WHU Building Dataset loader used by GeoRectNetPlus.

Based on the supplied GeoRectNet WHU notebook:
- RGB images
- binary building masks
- resize to cfg.img_size
- joint flip/rotation/photometric augmentation
- stable filename IDs for pseudo-label/TVR bookkeeping
"""

import glob
import os
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def load_images(folder: str):
    exts = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.bmp"]
    paths = []
    for ext in exts:
        paths += glob.glob(os.path.join(folder, ext))
    return sorted(paths)


class RandomSegDataset(Dataset):
    """Tiny random dataset used only as a fallback/smoke-test helper."""
    def __init__(self, length=10, img_size=512):
        self.length = length
        self.img_size = img_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = torch.rand(3, self.img_size, self.img_size)
        mask = (torch.rand(1, self.img_size, self.img_size) > 0.5).float()
        return img, mask


class WHUDataset(Dataset):
    """WHU Building Dataset with optional joint augmentation."""

    def __init__(
        self,
        image_paths,
        mask_paths=None,
        img_size=512,
        augment=False,
        return_id=False,
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.augment = augment
        self.return_id = return_id

    def __len__(self):
        return len(self.image_paths)

    def _augment(self, img, mask=None):
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if mask is not None:
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            if mask is not None:
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        k = random.randint(0, 3)
        if k == 1:
            img = img.transpose(Image.ROTATE_90)
            if mask is not None:
                mask = mask.transpose(Image.ROTATE_90)
        elif k == 2:
            img = img.transpose(Image.ROTATE_180)
            if mask is not None:
                mask = mask.transpose(Image.ROTATE_180)
        elif k == 3:
            img = img.transpose(Image.ROTATE_270)
            if mask is not None:
                mask = mask.transpose(Image.ROTATE_270)

        img_np = np.array(img).astype(np.float32)
        img_np = img_np * random.uniform(0.8, 1.2)

        mean_val = img_np.mean()
        img_np = (img_np - mean_val) * random.uniform(0.8, 1.2) + mean_val

        gray = img_np.mean(axis=2, keepdims=True)
        sat = random.uniform(0.7, 1.3)
        img_np = img_np * sat + gray * (1 - sat)

        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_np)

        return img, mask

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        img = Image.open(image_path).convert("RGB")
        img = img.resize(
            (self.img_size, self.img_size),
            resample=Image.BILINEAR,
        )

        mask = None
        if self.mask_paths is not None:
            mask = Image.open(self.mask_paths[idx]).convert("L")
            mask = mask.resize(
                (self.img_size, self.img_size),
                resample=Image.NEAREST,
            )

        if self.augment:
            img, mask = self._augment(img, mask)

        img = torch.tensor(
            np.array(img) / 255.0
        ).permute(2, 0, 1).float()

        if mask is not None:
            mask = (np.array(mask) > 127).astype(np.float32)
            mask = torch.tensor(mask).unsqueeze(0)

            if self.return_id:
                return img, mask, os.path.basename(image_path)

            return img, mask

        if self.return_id:
            return img, os.path.basename(image_path)

        return img
