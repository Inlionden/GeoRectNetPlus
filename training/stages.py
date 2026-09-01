"""GeoRectNetPlus three-stage training orchestration.

Matches the supplied notebook:
Warmup -> Stage 1 GT -> Stage 2 pseudo-label -> Stage 3 finetune.
"""

import torch


def make_full_optimizer(model, loss_fn, cfg):
    backbone_params = list(model.encoder.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    head_params = [
        p for p in model.parameters()
        if id(p) not in backbone_ids
    ]

    return torch.optim.AdamW([
        {"params": backbone_params, "lr": cfg.lr_backbone},
        {"params": head_params, "lr": cfg.lr_head},
        {"params": list(loss_fn.parameters()),
         "lr": cfg.lr_loss_weights},
    ])


def make_stage2_optimizer(model, loss_fn, cfg):
    head_params = [
        p for p in model.parameters() if p.requires_grad
    ]
    return torch.optim.AdamW([
        {"params": head_params, "lr": cfg.lr_head * 0.3},
        {"params": list(loss_fn.parameters()),
         "lr": cfg.lr_loss_weights},
    ])


def make_stage3_optimizer(model, loss_fn, cfg):
    backbone_params = list(model.encoder.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    head_params = [
        p for p in model.parameters()
        if id(p) not in backbone_ids
    ]

    return torch.optim.AdamW([
        {"params": backbone_params,
         "lr": cfg.lr_backbone * 0.5},
        {"params": head_params,
         "lr": cfg.lr_head * 0.5},
        {"params": list(loss_fn.parameters()),
         "lr": cfg.lr_loss_weights},
    ])


def freeze_backbone(model, frozen=True):
    for p in model.encoder.parameters():
        p.requires_grad = not frozen


def build_scheduler(optimizer, epochs):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs), eta_min=1e-7
    )


def create_amp_scaler(cfg):
    if cfg.use_amp and torch.cuda.is_available():
        return torch.amp.GradScaler("cuda")
    return None


def stage_plan(cfg):
    return [
        ("WARMUP", cfg.warmup_freeze),
        ("S1", cfg.epochs_stage1),
        ("S2", cfg.epochs_stage2),
        ("S3", cfg.epochs_stage3),
    ]
