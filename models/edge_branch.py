

class EdgeBranch(nn.Module):
    """Lightweight 3-layer CNN that produces a soft edge probability map.
    Input: 4 channels (RGB + HH wavelet).  Output: 1-channel logit map."""
    def __init__(self, in_ch: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32,    32, 3, padding=1)
        self.conv3 = nn.Conv2d(32,     1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.conv3(x)     # raw logit — caller applies sigmoid if needed

print("[cell2] image utilities defined")