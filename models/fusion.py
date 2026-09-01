

class MultiLayerFusion(nn.Module):
    """Fuse 4 ViT layers (each 768-ch @ 32×32) into a single feature map.
    Projects each layer to 256-ch, concatenates, then compresses to out_ch."""
    def __init__(self, embed_dim=768, out_ch=768):
        super().__init__()
        self.proj3  = nn.Sequential(nn.Conv2d(embed_dim, 256, 1), nn.BatchNorm2d(256), nn.ReLU(True))
        self.proj6  = nn.Sequential(nn.Conv2d(embed_dim, 256, 1), nn.BatchNorm2d(256), nn.ReLU(True))
        self.proj9  = nn.Sequential(nn.Conv2d(embed_dim, 256, 1), nn.BatchNorm2d(256), nn.ReLU(True))
        self.proj12 = nn.Sequential(nn.Conv2d(embed_dim, 256, 1), nn.BatchNorm2d(256), nn.ReLU(True))
        self.fuse = nn.Sequential(
            nn.Conv2d(256 * 4, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, multi_feats):
        p3  = self.proj3(multi_feats[0])
        p6  = self.proj6(multi_feats[1])
        p9  = self.proj9(multi_feats[2])
        p12 = self.proj12(multi_feats[3])
        return self.fuse(torch.cat([p3, p6, p9, p12], dim=1))

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling (DeepLabV3+ style).
    Captures multi-scale context at the bottleneck — buildings come in many sizes.
    Parallel dilated convolutions at rates 6, 12, 18 + global average pooling."""
    def __init__(self, in_ch, out_ch=256):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.atrous6 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.atrous12 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.atrous18 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.ReLU(inplace=True))
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * 5, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Dropout(0.1))

    def forward(self, x):
        h, w = x.shape[-2:]
        f1 = self.conv1x1(x)
        f2 = self.atrous6(x)
        f3 = self.atrous12(x)
        f4 = self.atrous18(x)
        f5 = self.gap(x)
        f5 = F.interpolate(f5, size=(h, w), mode="bilinear", align_corners=False)
        out = torch.cat([f1, f2, f3, f4, f5], dim=1)
        return self.project(out)