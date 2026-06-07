"""U-Net for super-resolution artifact segmentation.

Given a super-resolved image ``img`` (optionally concatenated with the
ground-truth reference ``gt``), predict a per-pixel probability mask of artifact
regions. The class name, ``threshold`` field and ``save_weights`` /
``load_weights`` methods follow the grading interface.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two 3x3 conv + BN + ReLU layers."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MyEncoder(nn.Module):
    """Downsampling path; returns the activation after each stage (for skips)."""

    def __init__(self, start_filters: int, num_blocks: int, use_gt: bool) -> None:
        super().__init__()
        self.first_block = DoubleConv(6 if use_gt else 3, start_filters)
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(DoubleConv(start_filters, start_filters * 2))
            start_filters *= 2
        self.bottleneck = DoubleConv(start_filters, start_filters * 2)

    def forward(self, x: torch.Tensor):
        activations = []
        x = self.first_block(x)
        activations.append(x)
        x = F.max_pool2d(x, 2)
        for block in self.blocks:
            x = block(x)
            activations.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        activations.append(x)
        return activations


class MyDecoderBlock(nn.Module):
    """Upsample, halve channels, concatenate the skip, and refine."""

    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.upconv = nn.Conv2d(out_channels * 2, out_channels, 3, padding=1)
        self.conv1 = nn.Conv2d(out_channels * 2, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, down: torch.Tensor, left: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(down, mode="bilinear", scale_factor=2, align_corners=False)
        x = self.upconv(x)
        x = torch.cat((left, x), dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x


class MyDecoder(nn.Module):
    """Upsampling path mirroring the encoder, consuming skip activations."""

    def __init__(self, input_channels: int, num_blocks: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        for idx in range(num_blocks):
            self.blocks.insert(0, MyDecoderBlock(input_channels * 2 ** idx))

    def forward(self, activations):
        output = activations[-1]
        for block, left in zip(self.blocks, activations[-2::-1]):
            output = block(output, left)
        return output


class MyModel(nn.Module):
    """Encoder-decoder artifact segmenter.

    :param use_gt: If ``True``, the ground-truth image is concatenated to the
        input (6 channels) so the model can compare SR output against reference.
    """

    def __init__(self, num_classes: int = 1, num_blocks: int = 2, start_filters: int = 16,
                 use_gt: bool = False, init: bool = False) -> None:
        super().__init__()
        self.encoder = MyEncoder(start_filters=start_filters, num_blocks=num_blocks, use_gt=use_gt)
        self.decoder = MyDecoder(input_channels=start_filters, num_blocks=num_blocks + 1)
        self.final = nn.Conv2d(start_filters, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.threshold = 0.75
        self.use_gt = use_gt
        if init:
            self.apply(self.init_weights)

    @staticmethod
    def init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: dict) -> torch.Tensor:
        img, gt = x["img"], x["gt"]
        unbatched = img.dim() == 3
        if unbatched:
            img, gt = img.unsqueeze(0), gt.unsqueeze(0)

        inp = torch.cat([img, gt], dim=1) if self.use_gt else img
        out = self.sigmoid(self.final(self.decoder(self.encoder(inp))))

        return out.squeeze(0) if unbatched else out

    def save_weights(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str, device=None) -> None:
        self.load_state_dict(torch.load(path, map_location=device))
