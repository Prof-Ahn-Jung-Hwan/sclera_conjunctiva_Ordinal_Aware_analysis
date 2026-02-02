import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath

from model.backbone import Bottleneck, create_backbone
from model.encoder import VariantionalEncoder
from model.head import Head


class HbNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.ordinal_bins = args.ordinal_bins
        self.use_encoder = args.use_encoder
        self.use_ordinal_regression = args.use_ordinal_regression

        self.backbone, self.out_dim = create_backbone(args)

        if (self.use_encoder or self.use_ordinal_regression) and (
            args.backbone.startswith("resnet")
            or args.backbone.startswith("resnext")
            or args.backbone.startswith("convnext")
        ):
            self.bottleneck = Bottleneck(self.out_dim)
            self.att_norm = nn.LayerNorm(self.out_dim)
            self.attn = nn.MultiheadAttention(self.out_dim, self.out_dim // 64, batch_first=True)

        if args.use_encoder:
            self.norm = nn.LayerNorm(self.out_dim)
            self.encoder = VariantionalEncoder(self.out_dim)
            z_prior = torch.empty(self.ordinal_bins - 1, self.out_dim)
            nn.init.orthogonal_(z_prior, gain=1.3)
            self.register_buffer("z_prior", z_prior)

        # Use DropPath for better regularization during training
        self.drop_path = DropPath(0.1)

        if args.use_ordinal_regression:
            self.head = Head(self.out_dim, (self.ordinal_bins - 1) * 2)
        else:
            self.head = nn.Linear(self.out_dim, 1)

    def forward(self, img, label=None, **kwargs):
        if self.use_ordinal_regression and self.training:
            assert label is not None

        z = self.backbone(img)
        if (self.use_encoder or self.use_ordinal_regression) and (
            self.args.backbone.startswith("resnet")
            or self.args.backbone.startswith("resnext")
            or self.args.backbone.startswith("convnext")
        ):
            z = self.bottleneck(z)
            z = rearrange(z, "b c h w -> b (h w) c")
            z = self.att_norm(z)
            t = torch.cat([z, z.mean(1, keepdims=True)], 1)
            t = self.attn(t, t, t, need_weights=False)[0]
            z = t[:, -1]
        else:
            if z.ndim == 4:
                z = F.adaptive_avg_pool2d(z, (1, 1)).flatten(1)
            elif z.ndim == 2:
                """Do nothing if the output is already flattened"""
            else:
                raise ValueError(f"Unsupported backbone output shape: {z.shape}")

        if self.use_encoder:
            z = self.norm(z)
            z = self.encoder(z)
        z = self.drop_path(z)

        if self.use_ordinal_regression:
            logits = self.head(z, label)
            logits = rearrange(logits, "B (K P) -> B K P", P=2)
            # ord_k is for inference/logging, gradient calculation is not needed
            with torch.no_grad():
                ord_k = F.relu(logits.softmax(-1).argmax(-1).sum(-1, keepdim=True) - 1).int()
        else:
            logits = self.head(z)
            ord_k = None

        return {
            "z": z,
            "logits": logits,
            "ord_k": ord_k,
            "z_prior": self.z_prior if self.use_encoder else None,
        }
