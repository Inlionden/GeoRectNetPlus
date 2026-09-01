"""WHU DataLoaders for GeoRectNetPlus."""

import os
from torch.utils.data import DataLoader

from .whu_dataset import WHUDataset, RandomSegDataset, load_images
from .split import FixedWHUSplit


def build_whu_dataloaders(
    data_root,
    img_size=512,
    label_frac=0.125,
    batch_size=4,
    num_workers=2,
    seed=42,
    smoke_test=False,
    smoke_max_train=20,
    smoke_max_val=8,
):
    train_img_dir = os.path.join(data_root, "train", "Image")
    train_mask_dir = os.path.join(data_root, "train", "Mask")
    val_img_dir = os.path.join(data_root, "val", "Image")
    val_mask_dir = os.path.join(data_root, "val", "Mask")

    train_imgs = load_images(train_img_dir)
    train_masks = load_images(train_mask_dir)
    val_imgs = load_images(val_img_dir)
    val_masks = load_images(val_mask_dir)

    n_train = min(len(train_imgs), len(train_masks))
    n_val = min(len(val_imgs), len(val_masks))

    train_imgs = train_imgs[:n_train]
    train_masks = train_masks[:n_train]
    val_imgs = val_imgs[:n_val]
    val_masks = val_masks[:n_val]

    if smoke_test:
        train_imgs = train_imgs[:smoke_max_train]
        train_masks = train_masks[:smoke_max_train]
        val_imgs = val_imgs[:smoke_max_val]
        val_masks = val_masks[:smoke_max_val]

    if not train_imgs:
        raise FileNotFoundError(
            f"No WHU training image/mask pairs found under {data_root}"
        )

    split = FixedWHUSplit(
        train_imgs,
        train_masks,
        label_frac=label_frac,
        seed=seed,
    )

    labeled_ds = WHUDataset(
        split.labeled_images,
        split.labeled_masks,
        img_size=img_size,
        augment=True,
        return_id=True,
    )
    unlabeled_ds = WHUDataset(
        split.unlabeled_images,
        None,
        img_size=img_size,
        augment=True,
        return_id=True,
    )

    val_ds = WHUDataset(
        val_imgs,
        val_masks,
        img_size=img_size,
        augment=False,
        return_id=False,
    )

    persistent = num_workers > 0

    labeled_loader = DataLoader(
        labeled_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
    )

    unlabeled_loader = DataLoader(
        unlabeled_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
    )

    return labeled_loader, unlabeled_loader, val_loader, split
