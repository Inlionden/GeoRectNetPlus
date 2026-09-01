"""GeoRectNetPlus model architecture.

Components are extracted from the supplied GeoRectNet WHU notebook.
This file contains the neural-network building blocks and DABLCNet model.
"""



def rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    """Luminance-weighted grayscale  [B,3,H,W] -> [B,1,H,W]."""
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b

def hh_wavelet_channel(x: torch.Tensor) -> torch.Tensor:
    """Haar HH (diagonal edge) channel  [B,3,H,W] -> [B,1,H,W].
    Highlights fine-grained boundary textures for the edge branch."""
    gray   = rgb_to_gray(x)
    kernel = torch.tensor([[1., -1.], [-1., 1.]],
                          device=gray.device).view(1, 1, 2, 2)
    hh = F.conv2d(gray, kernel, stride=1, padding=1)
    return hh[:, :, :x.shape[-2], :x.shape[-1]]

def kl_dirichlet(alpha: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    """KL divergence between Dir(α) and Dir(1,...,1)."""
    ones = torch.ones_like(alpha)
    S_alpha = alpha.sum(dim=1, keepdim=True)
    S_ones = ones.sum(dim=1, keepdim=True)
    kl = (torch.lgamma(S_alpha) - torch.lgamma(S_ones)
          - (torch.lgamma(alpha) - torch.lgamma(ones)).sum(dim=1, keepdim=True)
          + ((alpha - ones) * (torch.digamma(alpha) - torch.digamma(S_alpha))).sum(dim=1, keepdim=True))
    return kl.mean()

def edl_loss(alpha, target, epoch, total_epochs, annealing_epochs=10):
    """EDL loss = Bayes risk + annealed KL."""
    target_1h = torch.cat([1.0 - target, target], dim=1)
    S = alpha.sum(dim=1, keepdim=True)
    bayes_risk = (target_1h * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=1)
    annealing_coef = min(1.0, epoch / max(1, annealing_epochs))
    alpha_tilde = target_1h + (1.0 - target_1h) * (alpha - 1.0) + 1.0
    kl = kl_dirichlet(alpha_tilde, num_classes=2)
    return bayes_risk.mean() + annealing_coef * kl


# ═══════════════════════════════════════════════════════════════════
#  CONTRIBUTION 2: Cross-Layer Attention Agreement Map (CLAAM / MLAA)
# ═══════════════════════════════════════════════════════════════════

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

class ViTAttentionExtractor:
    """Capture ViT CLS-to-patch attention from selected transformer blocks."""
    def __init__(self, vit_encoder, layers=(2, 5, 8, 11), detach=True):
        self.layers = list(layers)
        self.detach = detach
        self._maps = {}
        self._hooks = []
        self._register(vit_encoder)

    def _register(self, vit_encoder):
        for idx in self.layers:
            block = vit_encoder.blocks[idx]
            hook = block.attn.register_forward_hook(self._make_hook(idx))
            self._hooks.append(hook)

    def _make_hook(self, layer_idx):
        def hook(module, input, output):
            x = input[0]
            B, N, C = x.shape
            qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, C // module.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k = qkv[0], qkv[1]
            scale = getattr(module, "scale", q.shape[-1] ** -0.5)
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            self._maps[layer_idx] = attn.detach() if self.detach else attn
        return hook

    def clear(self):
        self._maps = {}

    def get_spatial_maps(self, patch_h, patch_w):
        spatial = []
        for idx in self.layers:
            if idx not in self._maps:
                raise RuntimeError(f"Attention map for ViT layer {idx} was not captured")
            attn = self._maps[idx]
            cls_attn = attn[:, :, 0, 1:]
            expected = patch_h * patch_w
            if cls_attn.shape[-1] != expected:
                grid = int(math.sqrt(cls_attn.shape[-1]))
                if grid * grid != cls_attn.shape[-1]:
                    raise ValueError(f"Cannot reshape {cls_attn.shape[-1]} patch tokens into a grid")
                patch_h = patch_w = grid
            spatial.append(cls_attn.reshape(cls_attn.shape[0], cls_attn.shape[1], patch_h, patch_w))
        return spatial

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

class ViTEncoder(nn.Module):
    """512-resolution ViT: 512/16 = 32x32 tokens.
    Returns the same multi-layer feature maps as Phase-2, and stores
    attention maps for CAM-Guided CLAAM.
    """
    def __init__(self):
        super().__init__()
        if timm is None:
            raise RuntimeError("timm not available.")
        self.vit = timm.create_model(
            "vit_base_patch16_224", pretrained=True,
            num_classes=0, img_size=512)
        try:
            self.vit.set_grad_checkpointing(enable=True)
        except AttributeError:
            pass
        self.patch_size = 16
        self.embed_dim = 768
        self.hook_layers = [2, 5, 8, 11]
        self._features = {}
        self._last_attn_maps = None
        self._register_hooks()
        self.attn_extractor = ViTAttentionExtractor(self.vit, layers=self.hook_layers)

    def _register_hooks(self):
        for idx in self.hook_layers:
            block = self.vit.blocks[idx]
            block.register_forward_hook(self._make_hook(idx))

    def _make_hook(self, idx):
        def hook_fn(module, input, output):
            self._features[idx] = output
        return hook_fn

    def forward(self, x: torch.Tensor):
        mean = IMAGENET_MEAN.to(x.device)
        std = IMAGENET_STD.to(x.device)
        x = (x - mean) / std

        self._features = {}
        self.attn_extractor.clear()
        _ = self.vit.forward_features(x)

        b = x.shape[0]
        grid_h = x.shape[-2] // self.patch_size
        grid_w = x.shape[-1] // self.patch_size

        multi_feats = []
        for idx in self.hook_layers:
            feat = self._features[idx]
            if feat.shape[1] == grid_h * grid_w + 1:
                feat = feat[:, 1:, :]
            feat_map = feat.transpose(1, 2).reshape(b, self.embed_dim, grid_h, grid_w)
            multi_feats.append(feat_map)

        self._last_attn_maps = self.attn_extractor.get_spatial_maps(grid_h, grid_w)
        return multi_feats

    def get_attention_maps(self):
        if self._last_attn_maps is None:
            raise RuntimeError("Run the ViT encoder before requesting attention maps")
        return self._last_attn_maps

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

class ConfidenceHead(nn.Module):
    """Original sigmoid confidence head (baseline / fallback when EDL disabled)."""
    def __init__(self, in_ch: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 1, 1)
        self.log_temp = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        temp = torch.exp(self.log_temp) + 1e-6
        return torch.sigmoid(x / temp)


# ═══════════════════════════════════════════════════════════════════
#  CONTRIBUTION 1: Evidential Dirichlet Confidence Head (EDL-Conf)
# ═══════════════════════════════════════════════════════════════════

class EvidentialHead(nn.Module):
    """Outputs Dirichlet concentration params α = evidence + 1 per class.
    Epistemic uncertainty u = K/S where S = Σα, K = num_classes."""
    def __init__(self, in_ch: int = 32, num_classes: int = 2):
        super().__init__()
        self.K = num_classes
        self.conv1 = nn.Conv2d(in_ch, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, num_classes, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        evidence = F.softplus(self.conv3(x))
        alpha = evidence + 1.0
        S = alpha.sum(dim=1, keepdim=True)
        belief = evidence / S
        uncertainty = float(self.K) / S
        confidence = 1.0 - uncertainty
        return {
            "alpha": alpha,
            "belief": belief,
            "uncertainty": uncertainty,
            "confidence": confidence.clamp(0, 1),
            "strength": S,
        }


def kl_dirichlet(alpha: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    """KL divergence between Dir(α) and Dir(1,...,1)."""
    ones = torch.ones_like(alpha)
    S_alpha = alpha.sum(dim=1, keepdim=True)
    S_ones = ones.sum(dim=1, keepdim=True)
    kl = (torch.lgamma(S_alpha) - torch.lgamma(S_ones)
          - (torch.lgamma(alpha) - torch.lgamma(ones)).sum(dim=1, keepdim=True)
          + ((alpha - ones) * (torch.digamma(alpha) - torch.digamma(S_alpha))).sum(dim=1, keepdim=True))
    return kl.mean()


def edl_loss(alpha, target, epoch, total_epochs, annealing_epochs=10):
    """EDL loss = Bayes risk + annealed KL."""
    target_1h = torch.cat([1.0 - target, target], dim=1)
    S = alpha.sum(dim=1, keepdim=True)
    bayes_risk = (target_1h * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=1)
    annealing_coef = min(1.0, epoch / max(1, annealing_epochs))
    alpha_tilde = target_1h + (1.0 - target_1h) * (alpha - 1.0) + 1.0
    kl = kl_dirichlet(alpha_tilde, num_classes=2)
    return bayes_risk.mean() + annealing_coef * kl


# ═══════════════════════════════════════════════════════════════════
#  CONTRIBUTION 2: Cross-Layer Attention Agreement Map (CLAAM / MLAA)
# ═══════════════════════════════════════════════════════════════════

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

class DABLCNet(nn.Module):
    """DABLCNet + Phase-2 contributions with CAM-Guided CLAAM swapped in."""
    def __init__(self):
        super().__init__()

        # Core architecture
        self.encoder = ViTEncoder()
        self.layer_fuse = MultiLayerFusion(768, out_ch=768)
        self.aspp = ASPP(in_ch=768, out_ch=256)
        self.spatial_enc = SpatialEncoder()
        self.edge_branch = EdgeBranch(in_ch=4)
        self.decoder = EdgeGuidedDecoder(in_ch=256)

        # Contribution 1: Evidential Dirichlet head
        self.edl_head = EvidentialHead(in_ch=32, num_classes=2)

        # Contribution 2: CAM-Guided CLAAM
        self.claam = CAMGuidedCLAAM(embed_dim=768, num_heads=12, num_layers=4)

        # SDR: Signed Distance Regression head
        self.sdr_head = SignedDistanceHead(in_ch=32)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        hh = hh_wavelet_channel(x)
        edge_in = torch.cat([x, hh], dim=1)
        edge_map = self.edge_branch(edge_in)

        multi_feats = self.encoder(x)
        attn_maps = self.encoder.get_attention_maps()

        fused = self.layer_fuse(multi_feats)
        bottleneck = self.aspp(fused)
        skips = self.spatial_enc(x)
        logits = self.decoder(bottleneck, edge_map,
                              out_size=x.shape[-2:], skips=skips)

        dec_feat = self.decoder.get_features(bottleneck,
                                              out_size=x.shape[-2:], skips=skips)

        edl_out = self.edl_head(dec_feat)
        conf_map = edl_out["confidence"]

        claam_out = self.claam(attn_maps, multi_feats[-1], out_size=x.shape[-2:])
        sdf_pred = self.sdr_head(dec_feat)

        return {
            "logits": logits,
            "edge": edge_map,
            "conf": conf_map,
            "edl": edl_out,
            "claam": claam_out,
            "multi_feats": multi_feats,
            "sdf_pred": sdf_pred,
        }

print('[cell3] DABLCNet architecture defined')
