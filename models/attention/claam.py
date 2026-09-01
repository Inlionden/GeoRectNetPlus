"""CAM-Guided CLAAM and cross-layer consistency losses."""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

class CAMGuidedCLAAM(nn.Module):
    """v2: removes per-image min-max norm, adds per-layer projection
    and global tanh-gain. Output keys are unchanged."""
    def __init__(self, embed_dim: int = 768, num_heads: int = 12,
                 num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.cam_generator = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim // 4),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(embed_dim // 4, embed_dim // 16, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(embed_dim // 16),
            nn.GELU(),
            nn.Conv2d(embed_dim // 16, 1, kernel_size=1),
        )

        # NEW v2: per-layer projection forces specialization between layers
        # whose ViT features are nearly identical (DeepViT attention collapse).
        self.layer_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(num_heads, num_heads, kernel_size=1,
                          groups=num_heads, bias=False),
                nn.BatchNorm2d(num_heads),
                nn.GELU(),
            ) for _ in range(num_layers)
        ])

        self.head_weights = nn.Parameter(torch.ones(num_heads) / num_heads)
        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # Refine WITHOUT trailing Sigmoid; output gets gain-tanh below.
        self.refine = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )

        # NEW v2: global gain+bias REPLACE per-image min-max normalization.
        self.out_gain = nn.Parameter(torch.tensor(2.0))
        self.out_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, attn_maps_per_layer, feature_map, out_size):
        cam_logits = self.cam_generator(feature_map)
        cam = torch.sigmoid(cam_logits)

        projected = [self.layer_proj[i](a)
                     for i, a in enumerate(attn_maps_per_layer)]
        attn_stack = torch.stack(projected, dim=1)
        attn_cam = attn_stack * cam.unsqueeze(1)

        head_w = self.head_weights.softmax(dim=0)
        layer_w = self.layer_weights.softmax(dim=0)
        attn_hw = attn_cam * head_w.view(1, 1, -1, 1, 1)
        attn_lw = attn_hw * layer_w.view(1, -1, 1, 1, 1)

        layer_consensus = attn_lw.mean(dim=2)
        layer_std = layer_consensus.std(dim=1)
        layer_mean = layer_consensus.mean(dim=1)

        temp = self.temperature.clamp(min=0.1)
        raw_agreement = layer_mean / (1.0 + layer_std / temp)
        refined = self.refine(raw_agreement.unsqueeze(1))

        # KEY CHANGE: global tanh-gain instead of per-image min-max normalization
        agreement = 0.5 * (torch.tanh(self.out_gain * refined + self.out_bias) + 1.0)

        variance = layer_std.unsqueeze(1)
        mean_pred = layer_mean.unsqueeze(1)

        if agreement.shape[-2:] != out_size:
            agreement = F.interpolate(agreement, size=out_size, mode="bilinear", align_corners=False)
            variance = F.interpolate(variance, size=out_size, mode="bilinear", align_corners=False)
            mean_pred = F.interpolate(mean_pred, size=out_size, mode="bilinear", align_corners=False)
            cam = F.interpolate(cam, size=out_size, mode="bilinear", align_corners=False)
            cam_logits = F.interpolate(cam_logits, size=out_size, mode="bilinear", align_corners=False)

        return {
            "agreement": agreement,
            "variance": variance,
            "cam": cam,
            "cam_logits": cam_logits,
            "mean_pred": mean_pred,
            "head_weights": self.head_weights.softmax(dim=0),
            "layer_weights": self.layer_weights.softmax(dim=0),
        }

CLAAM = CAMGuidedCLAAM


def claam_consistency_loss(layer_logits, target):
    """Force ALL ViT layers to agree with GT."""
    loss = torch.tensor(0.0, device=target.device)
    for logit in layer_logits:
        loss = loss + F.binary_cross_entropy_with_logits(logit, target)
    return loss / len(layer_logits)


# ═══════════════════════════════════════════════════════════════════
#  IDEA 2: Cross-Layer Attention Consistency Loss (CLAC-Loss)
# ═══════════════════════════════════════════════════════════════════

def clac_loss(multi_feats: List[torch.Tensor]) -> torch.Tensor:
    """Penalizes spatial activation disagreement between adjacent ViT layers."""
    loss = torch.tensor(0.0, device=multi_feats[0].device)
    for i in range(len(multi_feats) - 1):
        act_i = multi_feats[i].mean(dim=1, keepdim=True)
        act_j = multi_feats[i + 1].mean(dim=1, keepdim=True)
        act_i = (act_i - act_i.amin(dim=(2, 3), keepdim=True)) / \
                (act_i.amax(dim=(2, 3), keepdim=True) - act_i.amin(dim=(2, 3), keepdim=True) + 1e-6)
        act_j = (act_j - act_j.amin(dim=(2, 3), keepdim=True)) / \
                (act_j.amax(dim=(2, 3), keepdim=True) - act_j.amin(dim=(2, 3), keepdim=True) + 1e-6)
        loss = loss + F.mse_loss(act_i, act_j)
    return loss / (len(multi_feats) - 1)


# ═══════════════════════════════════════════════════════════════════
#  IDEA 4: Gradient Orientation Prior (GOP)
# ═══════════════════════════════════════════════════════════════════

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

CLAAM = CAMGuidedCLAAM

def claam_consistency_loss(layer_logits, target):
    """Force ALL ViT layers to agree with GT."""
    loss = torch.tensor(0.0, device=target.device)
    for logit in layer_logits:
        loss = loss + F.binary_cross_entropy_with_logits(logit, target)
    return loss / len(layer_logits)


# ═══════════════════════════════════════════════════════════════════
#  IDEA 2: Cross-Layer Attention Consistency Loss (CLAC-Loss)
# ═══════════════════════════════════════════════════════════════════

def clac_loss(multi_feats: List[torch.Tensor]) -> torch.Tensor:
    """Penalizes spatial activation disagreement between adjacent ViT layers."""
    loss = torch.tensor(0.0, device=multi_feats[0].device)
    for i in range(len(multi_feats) - 1):
        act_i = multi_feats[i].mean(dim=1, keepdim=True)
        act_j = multi_feats[i + 1].mean(dim=1, keepdim=True)
        act_i = (act_i - act_i.amin(dim=(2, 3), keepdim=True)) / \
                (act_i.amax(dim=(2, 3), keepdim=True) - act_i.amin(dim=(2, 3), keepdim=True) + 1e-6)
        act_j = (act_j - act_j.amin(dim=(2, 3), keepdim=True)) / \
                (act_j.amax(dim=(2, 3), keepdim=True) - act_j.amin(dim=(2, 3), keepdim=True) + 1e-6)
        loss = loss + F.mse_loss(act_i, act_j)
    return loss / (len(multi_feats) - 1)


# ═══════════════════════════════════════════════════════════════════
#  IDEA 4: Gradient Orientation Prior (GOP)
# ═══════════════════════════════════════════════════════════════════