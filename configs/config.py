"""GeoRectNetPlus configuration.

Derived from the supplied GeoRectNet WHU notebook configuration.
Override dataset/output paths with environment variables when running locally.
"""

import os
from dataclasses import dataclass
import torch

SMOKE_TEST = os.getenv("GEORECT_SMOKE_TEST", "true").lower() in {"1", "true", "yes"}

_SM_TRAIN = 20
_SM_VAL = 8
_SM_PSEUDO = 5
_SM_EPOCHS = (1, 1, 1)
_SM_WARMUP = 1

_RT_EPOCHS = (15, 10, 5)
_RT_WARMUP = 3


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Config:
    # Dataset
    data_root: str = os.getenv(
        "WHU_ROOT",
        "/kaggle/input/datasets/sengulgs/whu-building-dataset/WHU",
    )
    img_size: int = 512

    # Primary paper setting: 1/8 = 12.5%
    label_frac: float = 0.01 if SMOKE_TEST else 0.125

    # Batch / workers
    batch_size: int = 2 if SMOKE_TEST else 4
    num_workers: int = 0 if SMOKE_TEST else 2

    # AMP + async gates
    use_amp: bool = True
    use_async_gates: bool = not SMOKE_TEST
    async_warmup_batches: int = 5 if SMOKE_TEST else 50

    # Epochs
    epochs_stage1: int = _SM_EPOCHS[0] if SMOKE_TEST else _RT_EPOCHS[0]
    epochs_stage2: int = _SM_EPOCHS[1] if SMOKE_TEST else _RT_EPOCHS[1]
    epochs_stage3: int = _SM_EPOCHS[2] if SMOKE_TEST else _RT_EPOCHS[2]
    warmup_freeze: int = _SM_WARMUP if SMOKE_TEST else _RT_WARMUP

    # Pseudo-label budget
    max_pseudo_batches: int = _SM_PSEUDO if SMOKE_TEST else 0
    pseudo_budget_N: int = 3

    # Smoke-test caps
    smoke_max_train: int = _SM_TRAIN
    smoke_max_val: int = _SM_VAL

    # Learning rates
    lr_backbone: float = 1e-5
    lr_head: float = 3e-4
    lr_loss_weights: float = 5e-5

    # Pseudo-label thresholds
    thr_start: float = 0.65
    thr_end: float = 0.50

    # Loss weights
    loss_w_in: float = 1.0
    loss_w_out: float = 1.5
    loss_dice_w: float = 1.0
    boundary_weight: float = 2.5
    conf_loss_weight: float = 0.3

    # Early stopping / saving
    patience: int = 3 if SMOKE_TEST else 12
    viz_every: int = 1 if SMOKE_TEST else 2
    pred_thr: float = 0.5
    preview_every: int = 0
    preview_samples: int = 2

    # Paths
    save_path: str = os.getenv(
        "GEORECT_SAVE_PATH", "checkpoints/best_model.pt"
    )
    results_dir: str = os.getenv(
        "GEORECT_RESULTS_DIR", "results"
    )
    checkpoint_path: str = os.getenv(
        "GEORECT_CHECKPOINT_PATH", "checkpoints/training_checkpoint.pt"
    )
    persist_dir: str = os.getenv(
        "GEORECT_PERSIST_DIR", "checkpoints/persist"
    )

    # EDL
    edl_loss_weight: float = 0.5
    edl_annealing_epochs: int = 2 if SMOKE_TEST else 10

    # CAM-Guided CLAAM v2
    claam_loss_weight: float = 0.4
    claam_cam_weight: float = 0.5
    claam_contrast_weight: float = 0.5
    claam_entropy_weight: float = 0.05
    claam_high_margin: float = 0.7
    claam_low_margin: float = 0.3
    claam_diversity_weight: float = 0.0
    claam_thr_start: float = 0.55
    claam_thr_end: float = 0.40
    cam_freeze_epochs: int = 0 if SMOKE_TEST else 5

    # CLAC
    clac_loss_weight: float = 0.2

    # TVR
    tvr_window: int = 5
    tvr_accept_ratio: float = 0.6

    # GOP
    gop_rectilinear_thr: float = 0.45

    # CAFCG
    cafcg_disagreement_thr: float = 0.3
    cafcg_keep_q: float = 0.70

    # SDR
    sdr_loss_weight: float = 0.3
    sdr_max_dist: float = 50.0

    # Pseudo-label misc
    pseudo_ramp_start: float = 0.1
    pseudo_freeze_backbone: bool = True
    pseudo_edge_erode_px: int = 5

    # Cross-dataset
    inria_root: str = os.getenv(
        "INRIA_ROOT",
        "/kaggle/input/inria-building/AerialImageDataset",
    )
    cross_dataset_img_size: int = 512

    log_every_n_batches: int = 0


cfg = Config()

os.makedirs(cfg.persist_dir, exist_ok=True)
os.makedirs(cfg.results_dir, exist_ok=True)
os.makedirs(os.path.dirname(cfg.save_path) or ".", exist_ok=True)
os.makedirs(os.path.dirname(cfg.checkpoint_path) or ".", exist_ok=True)


if __name__ == "__main__":
    print(f"SMOKE_TEST = {SMOKE_TEST}")
    print(
        f"label_frac = {cfg.label_frac} | "
        f"epochs S1/S2/S3 = "
        f"{cfg.epochs_stage1}/{cfg.epochs_stage2}/{cfg.epochs_stage3}"
    )
    print(f"batch = {cfg.batch_size}")
    print(f"data_root = {cfg.data_root}")
    print(f"save_path = {cfg.save_path}")
    print(f"results_dir = {cfg.results_dir}")
    print(f"device = {get_device()}")
