"""UNet architectures for map segmentation.

Mirrors the definitions used to train the shipped checkpoints, so that they
(raw ``state_dict`` ``.pth`` files, either nested
``encoder.*``/``decoder.*`` or legacy flat ``dconv_*`` keys) load unchanged.

Only imported when the ``unet`` strategy is selected, so torch stays an
optional dependency for classical-only deployments.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNetEncoder(nn.Module):
    """4-level encoder (same trunk as the multi-head UNet)."""

    def __init__(self) -> None:
        super().__init__()
        self.dconv_down1 = DoubleConv(3, 64)
        self.dconv_down2 = DoubleConv(64, 128)
        self.dconv_down3 = DoubleConv(128, 256)
        self.dconv_down4 = DoubleConv(256, 512)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        c1 = self.dconv_down1(x)
        c2 = self.dconv_down2(self.maxpool(c1))
        c3 = self.dconv_down3(self.maxpool(c2))
        c4 = self.dconv_down4(self.maxpool(c3))
        return c1, c2, c3, c4


def _legacy_to_nested(state_dict: dict) -> dict:
    """Map flat ``dconv_*`` / ``upsample`` keys to ``encoder.*`` / ``decoder.*``."""
    if any(k.startswith("encoder.") for k in state_dict):
        return state_dict
    out = {}
    for k, v in state_dict.items():
        if k.startswith("dconv_down") or k.startswith("maxpool"):
            out["encoder." + k] = v
        elif k.startswith("dconv_up") or k == "upsample" or k.startswith("upsample."):
            out["decoder." + k] = v
        else:
            out[k] = v
    return out


class UNetShallowDecoder(nn.Module):
    """3-level decoder: upsample from conv4 with skips (no 1024 bottleneck)."""

    def __init__(self) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dconv_up3 = DoubleConv(256 + 512, 256)
        self.dconv_up2 = DoubleConv(128 + 256, 128)
        self.dconv_up1 = DoubleConv(64 + 128, 64)

    def forward(
        self,
        conv1: torch.Tensor,
        conv2: torch.Tensor,
        conv3: torch.Tensor,
        conv4: torch.Tensor,
    ) -> torch.Tensor:
        x = self.upsample(conv4)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        return self.dconv_up1(x)


class UNet(nn.Module):
    """Single-head 4-class UNet: pixel in {background, red, yellow, roads}."""

    def __init__(self, n_classes: int = 4) -> None:
        super().__init__()
        self.encoder = UNetEncoder()
        self.decoder = UNetShallowDecoder()
        self.conv_last = nn.Conv2d(64, n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1, c2, c3, c4 = self.encoder(x)
        x = self.decoder(c1, c2, c3, c4)
        return self.conv_last(x)

    def load_state_dict(self, state_dict, strict: bool = True, **kwargs):
        return super().load_state_dict(_legacy_to_nested(dict(state_dict)), strict=strict, **kwargs)


class UNetMultiHeadDecoder(nn.Module):
    """4-level upsampling path with skip connections."""

    def __init__(self) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dconv_up4 = DoubleConv(512 + 1024, 512)
        self.dconv_up3 = DoubleConv(256 + 512, 256)
        self.dconv_up2 = DoubleConv(128 + 256, 128)
        self.dconv_up1 = DoubleConv(64 + 128, 64)

    def forward(
        self,
        x: torch.Tensor,
        conv1: torch.Tensor,
        conv2: torch.Tensor,
        conv3: torch.Tensor,
        conv4: torch.Tensor,
    ) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat([x, conv4], dim=1)
        x = self.dconv_up4(x)
        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        return self.dconv_up1(x)


class UNetMultiHead(nn.Module):
    """UNet with two output heads.

    - Road head: binary segmentation (background vs road)
    - Marking head: ternary segmentation (none vs red vs yellow)

    The marking head predictions are only meaningful on road pixels.
    """

    def __init__(self, n_marking_classes: int = 3) -> None:
        super().__init__()
        self.encoder = UNetEncoder()
        self.bottleneck = DoubleConv(512, 1024)
        self.decoder = UNetMultiHeadDecoder()
        self.road_head = nn.Conv2d(64, 2, 1)
        self.marking_head = nn.Conv2d(64, n_marking_classes, 1)

    def forward(self, x: torch.Tensor):
        conv1, conv2, conv3, conv4 = self.encoder(x)
        x = self.bottleneck(self.encoder.maxpool(conv4))
        features = self.decoder(x, conv1, conv2, conv3, conv4)
        return self.road_head(features), self.marking_head(features)

    def load_state_dict(self, state_dict, strict: bool = True, **kwargs):
        return super().load_state_dict(_legacy_to_nested(dict(state_dict)), strict=strict, **kwargs)
