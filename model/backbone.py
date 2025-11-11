import torch
import torch.nn as nn
from torchvision.models import (
    convnext_base,
    convnext_large,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnext50_32x4d,
    resnext101_64x4d,
    vit_b_16,
)


def create_backbone(args):
    if args.backbone.startswith("dino"):
        backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        out_dim = 768
        return backbone, out_dim

    try:
        backbone = eval(args.backbone)(weights="IMAGENET1K_V1")
    except:
        raise NotImplementedError(f"Backbone {args.backbone} is not supported!")

    if args.backbone.startswith("resnet") or args.backbone.startswith("resnext"):
        out_dim = backbone.inplanes
        backbone.layer4[0].conv2.stride = (1, 1)
        backbone.layer4[0].downsample[0].stride = (1, 1)
        modules = list(backbone.children())[:-2]
        backbone = nn.Sequential(*modules)
    elif args.backbone.startswith("convnext"):
        out_dim = backbone.classifier[0].normalized_shape[0]
        modules = list(backbone.children())[:-2]
        backbone = nn.Sequential(*modules)
    elif args.backbone.startswith("vit"):
        backbone.heads = nn.Identity()
        out_dim = backbone.hidden_dim
    else:
        raise NotImplementedError(f"Backbone {args.backbone} is not supported!")

    return backbone, out_dim


class Bottleneck(nn.Module):
    def __init__(self, channels, reduction=8, kernel_size=7):
        super().__init__()
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        ca = self.channel_attn(x)
        x = x * ca

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.spatial_attn(torch.cat([avg_out, max_out], dim=1))
        x = x * sa
        return x
