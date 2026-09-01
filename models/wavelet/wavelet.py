"""Image/wavelet utilities from GeoRectNetPlus Phase II."""

import torch
import torch.nn as nn
import torch.nn.functional as F

def rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    """Luminance-weighted grayscale  [B,3,H,W] -> [B,1,H,W]."""
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b


def hh_wavelet_channel(x: torch.Tensor) -> torch.Tensor:
    """Haar HH (diagonal edge) channel  [B,3,H,W] -> [B,1,H,W]."""
    gray   = rgb_to_gray(x)
    kernel = torch.tensor([[1., -1.], [-1., 1.]],
                          device=gray.device).view(1, 1, 2, 2)
    hh = F.conv2d(gray, kernel, stride=1, padding=1)
    return hh[:, :, :x.shape[-2], :x.shape[-1]]


class EdgeBranch(nn.Module):
    """Lightweight 3-layer CNN: RGB + HH wavelet -> edge logit map."""
    def __init__(self, in_ch: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.conv3(x)
