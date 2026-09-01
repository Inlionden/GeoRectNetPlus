"""Geometry and frequency/gradient gates from GeoRectNetPlus Phase II."""

import math
import numpy as np
import torch
import torch.nn.functional as F

def _geometry_gate(probs: torch.Tensor, thr: float = 0.5,
                   compact_min: float = 0.20, rect_min: float = 0.50,
                   max_aspect: float = 6.0, min_area: int = 20) -> torch.Tensor:
    """Geom v2: stricter rules - rejects tiny / non-rectangular / overly elongated comps."""
    import cv2
    B = probs.shape[0]
    gate = torch.ones_like(probs, dtype=torch.bool)
    for b in range(B):
        mask_np = (probs[b, 0].detach().cpu().numpy() > thr).astype(np.uint8) * 255
        if mask_np.sum() == 0:
            continue
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        reject_mask = np.zeros_like(mask_np, dtype=np.uint8)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                cv2.drawContours(reject_mask, [cnt], -1, 255, thickness=cv2.FILLED)
                continue
            peri = cv2.arcLength(cnt, True)
            compactness = (4 * math.pi * area) / (peri * peri + 1e-6)
            x, y, w, h = cv2.boundingRect(cnt)
            rectangularity = area / (w * h + 1e-6)
            aspect = max(w, h) / max(min(w, h), 1)
            if (compactness < compact_min or rectangularity < rect_min
                    or aspect > max_aspect):
                cv2.drawContours(reject_mask, [cnt], -1, 255, thickness=cv2.FILLED)
        if reject_mask.any():
            reject_t = torch.from_numpy(reject_mask > 0).to(probs.device)
            gate[b, 0] = gate[b, 0] & (~reject_t)
    return gate

def gradient_orientation_gate(probs: torch.Tensor, thr: float = 0.5,
                              rectilinear_thr: float = 0.45) -> torch.Tensor:
    """Reject pseudo-label components whose boundary gradient orientations
    deviate from rectilinear (0°/90°) angles."""
    import cv2
    B = probs.shape[0]
    gate = torch.ones_like(probs, dtype=torch.bool)

    for b in range(B):
        mask_np = (probs[b, 0].detach().cpu().numpy() > thr).astype(np.uint8) * 255
        if mask_np.sum() == 0:
            continue

        # AMP fix: probs may be float16 under autocast; cv2.Sobel needs float32/64.
        prob_np = probs[b, 0].detach().float().cpu().numpy()
        gx = cv2.Sobel(prob_np, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(prob_np, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        angle = np.degrees(np.arctan2(gy, gx)) % 180

        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        reject_mask = np.zeros_like(mask_np, dtype=np.uint8)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 30:
                continue
            comp_mask = np.zeros_like(mask_np)
            cv2.drawContours(comp_mask, [cnt], -1, 255, thickness=cv2.FILLED)
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(comp_mask, kernel, iterations=2)
            eroded = cv2.erode(comp_mask, kernel, iterations=1)
            boundary = ((dilated > 0) & ~(eroded > 0))

            boundary_mag = mag[boundary]
            boundary_angle = angle[boundary]
            if boundary_mag.size == 0:
                continue
            strong = boundary_mag > (boundary_mag.mean() + 1e-8)

            if strong.sum() < 5:
                continue

            angles_strong = boundary_angle[strong]
            rectilinear = (
                (angles_strong < 15) | (angles_strong > 165) |
                ((angles_strong > 75) & (angles_strong < 105))
            )
            rect_frac = rectilinear.sum() / len(angles_strong)

            if rect_frac < rectilinear_thr:
                cv2.drawContours(reject_mask, [cnt], -1, 255, thickness=cv2.FILLED)

        if reject_mask.any():
            reject_t = torch.from_numpy(reject_mask > 0).to(probs.device)
            gate[b, 0] = gate[b, 0] & (~reject_t)

    return gate


# ═══════════════════════════════════════════════════════════════════
#  IDEA 5a: Confidence-Aware Frequency Consistency Gate (CAFCG)
# ═══════════════════════════════════════════════════════════════════

def cafcg_gate(probs: torch.Tensor, disagreement_thr: float = None,
               keep_quantile: float = None) -> torch.Tensor:
    """CAFCG v2: ratio-based detector + adaptive per-image quantile threshold.
    The legacy `disagreement_thr` argument is kept for back-compat but ignored;
    use `keep_quantile` (or cfg.cafcg_keep_q) to control selectivity."""
    if keep_quantile is None:
        keep_quantile = getattr(cfg, "cafcg_keep_q", 0.70)
    B, C, H, W = probs.shape
    ksize, sigma = 9, 2.0
    x = torch.arange(ksize, device=probs.device).float() - ksize // 2
    g1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g1d = g1d / g1d.sum()
    g2d = (g1d.unsqueeze(1) * g1d.unsqueeze(0)).view(1, 1, ksize, ksize)
    low_freq = F.conv2d(probs, g2d, padding=ksize // 2)
    high_freq = (probs - low_freq).abs()
    ratio = high_freq / (low_freq.clamp_min(0.05))
    flat = ratio.flatten(2)
    thr = torch.quantile(flat, keep_quantile, dim=2, keepdim=True).unsqueeze(-1)
    return ratio <= thr
