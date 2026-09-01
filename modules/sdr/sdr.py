"""Signed Distance Regression utilities for GeoRectNetPlus."""

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import distance_transform_edt

class SignedDistanceHead(nn.Module):
    """Predicts normalized signed distance field (SDF) from decoder features.
    Output range: tanh -> [-1, +1] where +1 = deep interior, -1 = far outside,
    0 = on building boundary."""
    def __init__(self, in_ch: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        return torch.tanh(self.conv2(x))  # [-1, +1]


def compute_sdf_target(mask: torch.Tensor, max_dist: float = 50.0) -> torch.Tensor:
    """Compute normalized signed distance field from binary GT mask.
    Args:
        mask: [B, 1, H, W] binary building mask
        max_dist: clip and normalize distances to [-1, +1]
    Returns:
        sdf: [B, 1, H, W] where +1 = deep inside, -1 = far outside, 0 = boundary
    """
    from scipy.ndimage import distance_transform_edt
    B = mask.shape[0]
    sdf_batch = torch.zeros_like(mask)
    for b in range(B):
        m = mask[b, 0].cpu().numpy().astype(np.float64)
        if m.sum() > 0:
            dist_in = distance_transform_edt(m)
        else:
            dist_in = np.zeros_like(m)
        if (1 - m).sum() > 0:
            dist_out = distance_transform_edt(1 - m)
        else:
            dist_out = np.zeros_like(m)
        sdf = dist_in - dist_out  # + inside, - outside, 0 on boundary
        sdf = np.clip(sdf / max_dist, -1.0, 1.0)
        sdf_batch[b, 0] = torch.from_numpy(sdf).float()
    return sdf_batch.to(mask.device)


# ═══════════════════════════════════════════════════════════════════
#  Updated DABLCNet: integrates EDL, CLAAM, SDR, and all novel modules
# ═══════════════════════════════════════════════════════════════════

def compute_sdf_target(mask: torch.Tensor, max_dist: float = 50.0) -> torch.Tensor:
    """Compute normalized signed distance field from binary GT mask.
    Args:
        mask: [B, 1, H, W] binary building mask
        max_dist: clip and normalize distances to [-1, +1]
    Returns:
        sdf: [B, 1, H, W] where +1 = deep inside, -1 = far outside, 0 = boundary
    """
    from scipy.ndimage import distance_transform_edt
    B = mask.shape[0]
    sdf_batch = torch.zeros_like(mask)
    for b in range(B):
        m = mask[b, 0].cpu().numpy().astype(np.float64)
        if m.sum() > 0:
            dist_in = distance_transform_edt(m)
        else:
            dist_in = np.zeros_like(m)
        if (1 - m).sum() > 0:
            dist_out = distance_transform_edt(1 - m)
        else:
            dist_out = np.zeros_like(m)
        sdf = dist_in - dist_out  # + inside, - outside, 0 on boundary
        sdf = np.clip(sdf / max_dist, -1.0, 1.0)
        sdf_batch[b, 0] = torch.from_numpy(sdf).float()
    return sdf_batch.to(mask.device)


# ═══════════════════════════════════════════════════════════════════
#  Updated DABLCNet: integrates EDL, CLAAM, SDR, and all novel modules
# ═══════════════════════════════════════════════════════════════════
