"""Fixed labeled/unlabeled split for GeoRectNetPlus.

The supplied notebook fixes the split once using seed=42 so that the
labeled subset does not change from epoch to epoch.
"""

import random


class FixedWHUSplit:
    def __init__(self, image_paths, mask_paths, label_frac=0.125, seed=42):
        if len(image_paths) != len(mask_paths):
            n = min(len(image_paths), len(mask_paths))
            image_paths = image_paths[:n]
            mask_paths = mask_paths[:n]

        self.image_paths = list(image_paths)
        self.mask_paths = list(mask_paths)
        self.label_frac = label_frac
        self.seed = seed

        self.labeled_images = []
        self.labeled_masks = []
        self.unlabeled_images = []

        self._make_split()

    def _make_split(self):
        n = len(self.image_paths)
        if n == 0:
            return

        n_labeled = max(1, int(n * self.label_frac))

        rng = random.Random(self.seed)
        indices = list(range(n))
        rng.shuffle(indices)

        lab_idx = indices[:n_labeled]
        unl_idx = indices[n_labeled:]

        self.labeled_images = [self.image_paths[i] for i in lab_idx]
        self.labeled_masks = [self.mask_paths[i] for i in lab_idx]
        self.unlabeled_images = [self.image_paths[i] for i in unl_idx]

    @property
    def labeled_fraction(self):
        if not self.image_paths:
            return 0.0
        return len(self.labeled_images) / len(self.image_paths)

    def summary(self):
        return {
            "total": len(self.image_paths),
            "labeled": len(self.labeled_images),
            "unlabeled": len(self.unlabeled_images),
            "fraction": self.labeled_fraction,
            "seed": self.seed,
        }
