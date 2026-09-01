"""ViT backbone and attention extraction for GeoRectNetPlus."""

import math
import torch
import torch.nn as nn
import timm

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

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