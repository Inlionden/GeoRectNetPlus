

class SpatialEncoder(nn.Module):
    """Lightweight CNN for multi-scale spatial features -> decoder skip connections.
    Gives the decoder high-res boundary info that ViT's 32×32 bottleneck loses.
    Only ~160K params — negligible vs ViT's 86M."""
    def __init__(self):
        super().__init__()
        # 512 -> 256, 32ch
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        # 256 -> 128, 64ch
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))

    def forward(self, x):
        s1 = self.layer1(x)   # [B, 32, H/2, W/2]  = 256
        s2 = self.layer2(s1)  # [B, 64, H/4, W/4]  = 128
        return s1, s2

class EdgeGuidedDecoder(nn.Module):
    """Decoder for 32×32 bottleneck -> 512. All learned upsampling — no bilinear blur.
    32->64->128(+skip@128)->256(+skip@256)->512 with edge fusion."""
    def __init__(self, in_ch: int = 256):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_ch, 256, 2, stride=2),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.skip_fuse2 = nn.Sequential(
            nn.Conv2d(192, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.skip_fuse3 = nn.Sequential(
            nn.Conv2d(96, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.refine = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.fuse = nn.Sequential(
            nn.Conv2d(32 + 1, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.out_conv = nn.Conv2d(32, 1, 1)

    def _decode(self, feat, out_size, skips=None):
        """Shared decode path: 32->64->128->256->512."""
        x = self.up1(feat)
        x = self.up2(x)
        if skips is not None:
            s2 = F.interpolate(skips[1], size=x.shape[-2:],
                               mode="bilinear", align_corners=False)
            x = self.skip_fuse2(torch.cat([x, s2], dim=1))
        x = self.up3(x)
        if skips is not None:
            s1 = F.interpolate(skips[0], size=x.shape[-2:],
                               mode="bilinear", align_corners=False)
            x = self.skip_fuse3(torch.cat([x, s1], dim=1))
        x = self.up4(x)
        if x.shape[-2] != out_size[0] or x.shape[-1] != out_size[1]:
            x = F.interpolate(x, size=out_size, mode="nearest")
        x = self.refine(x)
        return x

    def forward(self, feat: torch.Tensor, edge: torch.Tensor,
                out_size: Tuple[int, int], skips=None) -> torch.Tensor:
        x = self._decode(feat, out_size, skips)
        edge_resized = F.interpolate(edge, size=out_size, mode="nearest")
        x = torch.cat([x, edge_resized], dim=1)
        x = self.fuse(x)
        return self.out_conv(x)

    def get_features(self, feat: torch.Tensor, out_size: Tuple[int, int],
                     skips=None) -> torch.Tensor:
        """Return 32-ch feature map for confidence head (no edge fusion)."""
        return self._decode(feat, out_size, skips)