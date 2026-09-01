"""GeoRectNetPlus training utilities.

Based on the supplied GeoRectNet WHU notebook Cell 9:
- AMP training
- supervised and pseudo-label epochs
- validation metrics
- async pseudo-label cache
- CSV logging
- crash-safe checkpoints
- best-model saving
"""

import csv
import gc
import hashlib
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi


def progressive_threshold(epoch, total_epochs, start, end):
    if total_epochs <= 1:
        return end
    t = epoch / (total_epochs - 1)
    return start + t * (end - start)


def _make_image_ids(imgs):
    ids = []
    for i in range(imgs.shape[0]):
        small = F.interpolate(
            imgs[i:i+1], size=(32, 32),
            mode="bilinear", align_corners=False
        )
        h = hashlib.md5(small.detach().cpu().numpy().tobytes()).hexdigest()[:12]
        ids.append(h)
    return ids


class AsyncGateCache:
    """Background CPU pseudo-label generation with bounded LRU caching."""

    def __init__(self, gate_fn, max_inflight=2, max_entries=1500):
        self.gate_fn = gate_fn
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.cache = OrderedDict()
        self.lock = threading.Lock()
        self.in_flight = set()
        self.max_inflight = max_inflight
        self.max_entries = max_entries
        self.stats = {
            "submitted": 0, "completed": 0, "hits": 0,
            "misses": 0, "evicted": 0
        }

    def submit(self, probs_cpu, model_out_cpu, image_ids,
               epoch, total_epochs, thr):
        new_idx, new_ids = [], []
        with self.lock:
            for i, iid in enumerate(image_ids):
                if iid not in self.in_flight:
                    self.in_flight.add(iid)
                    new_idx.append(i)
                    new_ids.append(iid)

        if not new_ids:
            return

        probs_sub = probs_cpu[new_idx]
        out_sub = {
            "edge": model_out_cpu.get("edge")[new_idx],
            "edl": {
                "uncertainty":
                model_out_cpu["edl"]["uncertainty"][new_idx]
            },
            "claam": {
                "agreement":
                model_out_cpu["claam"]["agreement"][new_idx]
            },
        }

        self.executor.submit(
            self._compute, probs_sub, out_sub,
            new_ids, epoch, total_epochs, thr
        )
        self.stats["submitted"] += len(new_ids)

    def _compute(self, probs, model_out, image_ids,
                 epoch, total_epochs, thr):
        try:
            pseudo = self.gate_fn(
                probs, model_out, thr, epoch,
                total_epochs, image_ids=image_ids
            )
            with self.lock:
                for i, iid in enumerate(image_ids):
                    if iid in self.cache:
                        del self.cache[iid]
                    self.cache[iid] = pseudo[i:i+1].detach().cpu()
                    self.in_flight.discard(iid)
                    self.stats["completed"] += 1

                while len(self.cache) > self.max_entries:
                    self.cache.popitem(last=False)
                    self.stats["evicted"] += 1
        except Exception as exc:
            with self.lock:
                for iid in image_ids:
                    self.in_flight.discard(iid)
            print(f"[async gate error] {type(exc).__name__}: {exc}")

    def get(self, image_ids):
        with self.lock:
            result = []
            for iid in image_ids:
                if iid not in self.cache:
                    self.stats["misses"] += 1
                    return None
                self.cache.move_to_end(iid)
                result.append(self.cache[iid])
            self.stats["hits"] += 1
        return torch.cat(result, dim=0)

    def shutdown(self):
        self.executor.shutdown(wait=True, cancel_futures=False)


def _model_output_to_cpu(out):
    result = {}
    if out.get("edge") is not None:
        result["edge"] = out["edge"].detach().float().cpu()

    edl = out.get("edl") or {}
    if edl.get("uncertainty") is not None:
        result["edl"] = {
            "uncertainty":
            edl["uncertainty"].detach().float().cpu()
        }

    claam = out.get("claam") or {}
    if claam.get("agreement") is not None:
        result["claam"] = {
            "agreement":
            claam["agreement"].detach().float().cpu()
        }

    return result


class TrainingLogger:
    """Creates the CSV logs used by the supplied notebook."""

    def __init__(self, results_dir):
        self.dir = Path(results_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

        self.train_csv = self.dir / "train_log.csv"
        self.gate_csv = self.dir / "gate_log.csv"
        self.loss_csv = self.dir / "loss_components.csv"
        self.claam_csv = self.dir / "claam_health.csv"
        self.weights_csv = self.dir / "weights_log.csv"

        self._init_csv(
            self.train_csv,
            ["global_epoch", "stage", "stage_epoch",
             "train_loss", "val_iou", "val_biou",
             "val_dice", "val_ece", "val_precision",
             "val_recall", "lr", "elapsed_sec"]
        )
        self._init_csv(
            self.gate_csv,
            ["global_epoch", "stage", "prob_pass",
             "edl_pass", "edge_pass", "geom_pass",
             "claam_pass", "tvr_pass", "gop_pass",
             "cafcg_pass", "final_pass_rate"]
        )
        self._init_csv(
            self.loss_csv,
            ["global_epoch", "stage", "bce", "dice",
             "boundary", "edl", "claam_cam",
             "claam_contrast", "claam_entropy",
             "clac", "sdr", "total"]
        )
        self._init_csv(
            self.claam_csv,
            ["global_epoch", "stage",
             "building_agreement",
             "background_agreement",
             "gap", "spatial_std"]
        )
        self._init_csv(
            self.weights_csv,
            ["global_epoch", "stage", "w_in", "w_out"]
        )

    @staticmethod
    def _init_csv(path, header):
        if not path.exists():
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(header)

    @staticmethod
    def append(csv_path, row):
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)


def save_checkpoint(state_dict, path):
    """Crash-safe atomic checkpoint."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(state_dict, tmp)
    if os.path.exists(path):
        os.replace(tmp, path)
    else:
        os.rename(tmp, path)


def load_checkpoint_if_exists(path):
    if not os.path.exists(path):
        return None
    try:
        return torch.load(path, map_location="cpu")
    except Exception as exc:
        print(f"[checkpoint] failed to load: {exc}")
        return None


def save_best_model(model, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state_cpu = {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
    }
    torch.save(state_cpu, path)
    del state_cpu
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return path


@torch.no_grad()
def validate_full(model, loader, device, metric_fns):
    """Returns BIoU, IoU, Dice, ECE, Precision and Recall.

    metric_fns must provide:
        boundary_iou, ece_score
    """
    model.eval()
    biou_a, iou_a, ece_a = [], [], []
    dice_a, prec_a, rec_a = [], [], []

    for batch in loader:
        imgs, masks = batch[0].to(device), batch[1].to(device)
        probs = torch.sigmoid(model(imgs)["logits"])
        pred = (probs >= 0.5).float()

        eps = 1e-6
        tp = (pred * masks).sum(dim=(2, 3))
        fp = (pred * (1 - masks)).sum(dim=(2, 3))
        fn = ((1 - pred) * masks).sum(dim=(2, 3))

        prec_a.append((tp / (tp + fp + eps)).mean().item())
        rec_a.append((tp / (tp + fn + eps)).mean().item())

        inter = (pred * masks).sum(dim=(2, 3))
        union = (
            pred.sum(dim=(2, 3)) +
            masks.sum(dim=(2, 3)) -
            inter
        )
        iou_a.append((inter / (union + eps)).mean().item())
        dice_a.append(
            (
                2 * inter /
                (pred.sum(dim=(2, 3)) +
                 masks.sum(dim=(2, 3)) + eps)
            ).mean().item()
        )

        biou_a.append(
            metric_fns["boundary_iou"](probs, masks).item()
        )
        ece_a.append(
            metric_fns["ece_score"](probs, masks).item()
        )

    if not iou_a:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    return (
        float(np.mean(biou_a)),
        float(np.mean(iou_a)),
        float(np.mean(dice_a)),
        float(np.mean(ece_a)),
        float(np.mean(prec_a)),
        float(np.mean(rec_a)),
    )


def train_one_epoch(
    model,
    labeled_loader,
    optimizer,
    loss_fn,
    epoch,
    total_epochs,
    device,
    cfg,
    pseudo_gate_fn=None,
    unlabeled_loader=None,
    scaler=None,
    async_gate_cache=None,
):
    """Train one epoch using the notebook's supervised/pseudo stages."""

    model.train()
    total_loss = 0.0
    steps = 0

    comp_acc = {
        "bce": 0.0, "dice": 0.0, "boundary": 0.0,
        "edl": 0.0, "claam_cam": 0.0,
        "claam_contrast": 0.0, "claam_entropy": 0.0,
        "clac": 0.0, "sdr": 0.0, "total": 0.0,
    }
    n_log = 0

    thr = progressive_threshold(
        epoch, total_epochs,
        cfg.thr_start, cfg.thr_end
    )
    pseudo_ramp = min(
        1.0,
        cfg.pseudo_ramp_start +
        (1.0 - cfg.pseudo_ramp_start) *
        epoch / max(total_epochs - 1, 1)
    )

    use_amp = cfg.use_amp and torch.cuda.is_available()

    def autocast_context():
        if use_amp:
            return torch.amp.autocast("cuda")
        from contextlib import nullcontext
        return nullcontext()

    # Pseudo-label training
    if unlabeled_loader is not None and pseudo_gate_fn is not None:
        if cfg.max_pseudo_batches > 0:
            pseudo_limit = cfg.max_pseudo_batches
        else:
            budget_imgs = (
                cfg.pseudo_budget_N *
                max(1, len(getattr(cfg, "_LABELED_IMG", [])))
            )
            pseudo_limit = min(
                len(unlabeled_loader),
                max(1, budget_imgs // cfg.batch_size)
            )

        for batch_idx, batch in enumerate(unlabeled_loader):
            if batch_idx >= pseudo_limit:
                break

            imgs = batch[0].to(device, non_blocking=True)
            optimizer.zero_grad()

            with autocast_context():
                out = model(imgs)
                probs = torch.sigmoid(out["logits"])

            image_ids = (
                list(batch[2]) if len(batch) >= 3
                and not torch.is_tensor(batch[2])
                else _make_image_ids(imgs)
            )

            target = None
            warmup = batch_idx < cfg.async_warmup_batches

            if (
                cfg.use_async_gates
                and async_gate_cache is not None
                and not warmup
            ):
                probs_cpu = probs.detach().float().cpu()
                out_cpu = _model_output_to_cpu(out)
                async_gate_cache.submit(
                    probs_cpu, out_cpu, image_ids,
                    epoch, total_epochs, thr
                )
                cached = async_gate_cache.get(image_ids)
                if cached is not None:
                    target = cached.to(device, non_blocking=True)

            if target is None:
                target = pseudo_gate_fn(
                    probs, out, thr, epoch,
                    total_epochs, image_ids=image_ids
                )

            if target.sum() < 1.0:
                continue

            with autocast_context():
                loss = loss_fn(
                    out["logits"], target, out.get("conf"),
                    model_output=out, epoch=epoch,
                    total_epochs=total_epochs,
                    is_pseudo=True
                )
                loss = loss * pseudo_ramp

            if not torch.isfinite(loss):
                continue

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) +
                    list(loss_fn.parameters()), max_norm=2.0
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) +
                    list(loss_fn.parameters()), max_norm=2.0
                )
                optimizer.step()

            total_loss += loss.item()
            steps += 1

            if hasattr(loss_fn, "last_components"):
                for k, v in loss_fn.last_components.items():
                    if k in comp_acc:
                        comp_acc[k] += v
                comp_acc["total"] += loss.item()
                n_log += 1

    # Supervised labeled training
    for batch in labeled_loader:
        imgs, masks = batch[0], batch[1]
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad()

        with autocast_context():
            out = model(imgs)
            loss = loss_fn(
                out["logits"], masks, out.get("conf"),
                model_output=out, epoch=epoch,
                total_epochs=total_epochs,
                is_pseudo=False
            )

        if not torch.isfinite(loss):
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) +
                list(loss_fn.parameters()), max_norm=2.0
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) +
                list(loss_fn.parameters()), max_norm=2.0
            )
            optimizer.step()

        total_loss += loss.item()
        steps += 1

        if hasattr(loss_fn, "last_components"):
            for k, v in loss_fn.last_components.items():
                if k in comp_acc:
                    comp_acc[k] += v
            comp_acc["total"] += loss.item()
            n_log += 1

    if n_log:
        for k in comp_acc:
            comp_acc[k] /= n_log

    return total_loss / max(1, steps), comp_acc


def run_stage(
    model, optimizer, scheduler, loss_fn, labeled_loader,
    unlabeled_loader, val_loader, total_epochs, stage_name,
    start_epoch, best_biou, device, cfg, metric_fns,
    pseudo_gate_fn=None, scaler=None, logger=None,
    async_gate_cache=None
):
    """Run one stage with validation, best-model saving and checkpoints."""

    no_improve = 0
    for epoch in range(start_epoch, total_epochs):
        t0 = time.time()

        use_pseudo = (
            stage_name in {"S2", "STAGE2"} and
            unlabeled_loader is not None
        )

        loss, components = train_one_epoch(
            model=model,
            labeled_loader=labeled_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epoch=epoch,
            total_epochs=total_epochs,
            device=device,
            cfg=cfg,
            pseudo_gate_fn=pseudo_gate_fn if use_pseudo else None,
            unlabeled_loader=unlabeled_loader if use_pseudo else None,
            scaler=scaler,
            async_gate_cache=async_gate_cache,
        )

        biou, iou, dice, ece, prec, rec = validate_full(
            model, val_loader, device, metric_fns
        )
        elapsed = time.time() - t0

        print(
            f"{stage_name} e{epoch}: "
            f"loss={loss:.4f} IoU={iou:.4f} "
            f"Dice={dice:.4f} BIoU={biou:.4f} "
            f"ECE={ece:.4f} ({elapsed:.1f}s)"
        )

        w_in = (
            loss_fn.w_in.item()
            if hasattr(loss_fn, "w_in") else None
        )
        w_out = (
            loss_fn.w_out.item()
            if hasattr(loss_fn, "w_out") else None
        )

        if logger is not None:
            lr = optimizer.param_groups[0]["lr"]
            logger.append(
                logger.train_csv,
                [epoch, stage_name, epoch,
                 f"{loss:.6f}", f"{iou:.6f}",
                 f"{biou:.6f}", f"{dice:.6f}",
                 f"{ece:.6f}", f"{prec:.6f}",
                 f"{rec:.6f}", f"{lr:.2e}",
                 f"{elapsed:.1f}"]
            )

            keys = [
                "bce", "dice", "boundary", "edl",
                "claam_cam", "claam_contrast",
                "claam_entropy", "clac", "sdr", "total"
            ]
            logger.append(
                logger.loss_csv,
                [epoch, stage_name] +
                [f"{components.get(k, 0.0):.6f}" for k in keys]
            )

            if w_in is not None:
                logger.append(
                    logger.weights_csv,
                    [epoch, stage_name,
                     f"{w_in:.6f}", f"{w_out:.6f}"]
                )

        if biou > best_biou:
            best_biou = biou
            save_best_model(model, cfg.save_path)
            no_improve = 0
            print(f"  saved best model -> {cfg.save_path}")
        else:
            no_improve += 1

        state = {
            "stage": stage_name,
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": (
                scheduler.state_dict()
                if scheduler is not None else None
            ),
            "scaler_state": (
                scaler.state_dict()
                if scaler is not None else None
            ),
            "best_biou": best_biou,
            "no_improve": no_improve,
        }
        save_checkpoint(state, cfg.checkpoint_path)

        if scheduler is not None:
            scheduler.step()

        if no_improve >= cfg.patience:
            print(f"Early stopping at {stage_name} epoch {epoch}")
            break

    return best_biou
