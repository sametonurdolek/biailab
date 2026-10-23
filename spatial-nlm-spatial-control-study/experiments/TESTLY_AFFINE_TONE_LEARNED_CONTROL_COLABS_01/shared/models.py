"""Control networks. They emit tone controls, never pixels.

The full backbone is the V2 Testly FPN encoder so an optional warm start can
load the epoch-54 encoder. A lightweight backbone is a separate config.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

MODEL_INPUT = 512


def gn(ch: int) -> nn.GroupNorm:
    groups = next(g for g in (16, 8, 4, 2, 1) if ch % g == 0)
    return nn.GroupNorm(groups, ch)


class ConvGNAct(nn.Module):
    def __init__(self, cin, cout, stride=1, kernel=3):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel, stride, kernel // 2, bias=False)
        self.norm = gn(cout)

    def forward(self, x):
        return F.silu(self.norm(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.norm1 = gn(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.norm2 = gn(ch)

    def forward(self, x):
        return F.silu(x + self.norm2(self.conv2(F.silu(self.norm1(self.conv1(x))))))


class EncoderV2(nn.Module):
    """Same stem and stages as TestlyFPN128AbsoluteGate's encoder."""

    def __init__(self, cin: int = 1):
        super().__init__()
        self.stem = ConvGNAct(cin, 32, 2)
        self.stage1 = nn.Sequential(ResBlock(32), ResBlock(32))
        self.down2 = ConvGNAct(32, 64, 2)
        self.stage2 = nn.Sequential(ResBlock(64), ResBlock(64))
        self.down3 = ConvGNAct(64, 96, 2)
        self.stage3 = nn.Sequential(ResBlock(96), ResBlock(96))
        self.down4 = ConvGNAct(96, 160, 2)
        self.stage4 = nn.Sequential(ResBlock(160), ResBlock(160))

    def features(self, x):
        f1 = self.stage1(self.stem(x))
        f2 = self.stage2(self.down2(f1))
        f3 = self.stage3(self.down3(f2))
        f4 = self.stage4(self.down4(f3))
        return f1, f2, f3, f4


class EncoderLight(nn.Module):
    def __init__(self, cin: int = 1):
        super().__init__()
        self.stem = ConvGNAct(cin, 16, 2)
        self.stage1 = nn.Sequential(ResBlock(16))
        self.down2 = ConvGNAct(16, 32, 2)
        self.stage2 = nn.Sequential(ResBlock(32))
        self.down3 = ConvGNAct(32, 48, 2)
        self.stage3 = nn.Sequential(ResBlock(48))
        self.down4 = ConvGNAct(48, 64, 2)
        self.stage4 = nn.Sequential(ResBlock(64))

    def features(self, x):
        f1 = self.stage1(self.stem(x))
        f2 = self.stage2(self.down2(f1))
        f3 = self.stage3(self.down3(f2))
        f4 = self.stage4(self.down4(f3))
        return f1, f2, f3, f4


def masked_pool(feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = F.interpolate(mask.float(), size=feat.shape[-2:], mode="nearest")
    return (feat.float() * m).sum((2, 3)) / m.sum((2, 3)).clamp_min(1)


def page_sample_grid(geom: torch.Tensor, grid: int) -> torch.Tensor:
    """512-image edge coordinates of the GxG cell centers, in grid_sample order."""
    g = geom.to(dtype=torch.float32)
    top, left, nh, nw = g.unbind(1)
    centers = (torch.arange(grid, device=g.device, dtype=torch.float32) + 0.5) / grid
    yy = top[:, None] + nh[:, None] * centers
    xx = left[:, None] + nw[:, None] * centers
    y = yy[:, :, None].expand(-1, -1, grid)
    x = xx[:, None, :].expand(-1, grid, -1)
    return torch.stack((2 * x / MODEL_INPUT - 1, 2 * y / MODEL_INPUT - 1), -1)


class ToneController(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, grid: int, lightweight: bool = False, kind: str = "spatial"):
        super().__init__()
        if in_ch not in (1, 2) or out_ch < 1 or grid < 1:
            raise RuntimeError("bad controller shape")
        self.kind = kind
        self.grid = int(grid)
        self.out_ch = int(out_ch)
        self.in_ch = int(in_ch)
        self.lightweight = bool(lightweight)
        self.encoder = EncoderLight(in_ch) if lightweight else EncoderV2(in_ch)
        channels = (16, 32, 48, 64) if lightweight else (32, 64, 96, 160)
        lateral_ch = 24 if lightweight else 48
        if kind == "global":
            self.head = nn.Sequential(
                nn.Linear(sum(channels), 64),
                nn.SiLU(),
                nn.Linear(64, out_ch),
            )
            return
        self.lateral = nn.ModuleList([ConvGNAct(c, lateral_ch, kernel=1) for c in channels])
        self.smooth = nn.ModuleList([ConvGNAct(lateral_ch, lateral_ch) for _ in range(4)])
        self.global_projection = nn.Linear(sum(channels), 32)
        self.context = ConvGNAct(lateral_ch + 32, lateral_ch)
        self.classifier = nn.Conv2d(lateral_ch, out_ch, 1, bias=True)
        nn.init.normal_(self.classifier.weight, std=0.001)
        nn.init.zeros_(self.classifier.bias)

    def set_stage(self, stage: str) -> None:
        self.requires_grad_(True)
        if stage == "full":
            return
        if stage == "head":
            self.encoder.requires_grad_(False)
            return
        if stage == "upper":
            # Deep stages only, matching the V2 stage-2 encoder thaw (stage3 and stage4).
            for name, param in self.encoder.named_parameters():
                if not (name.startswith("stage3") or name.startswith("stage4")):
                    param.requires_grad_(False)
            return
        raise RuntimeError(stage)

    def forward(self, x, mask, geom):
        frozen = not any(param.requires_grad for param in self.encoder.parameters())
        if frozen:
            with torch.no_grad():
                features = self.encoder.features(x)
        else:
            features = self.encoder.features(x)
        if self.kind == "global":
            desc = torch.cat([masked_pool(f, mask) for f in features], 1)
            return self.head(desc).float()
        laterals = [layer(feat) for layer, feat in zip(self.lateral, features)]
        pyramid = self.smooth[3](laterals[3])
        for index in (2, 1, 0):
            up = F.interpolate(pyramid, size=laterals[index].shape[-2:], mode="nearest")
            pyramid = self.smooth[index](laterals[index] + up)
        desc = torch.cat([masked_pool(f, mask) for f in features], 1)
        context = F.silu(self.global_projection(desc)).float()
        grid = page_sample_grid(geom, self.grid)
        sampled = F.grid_sample(
            pyramid.float(), grid, mode="bilinear", padding_mode="border", align_corners=False
        )
        context = context[:, :, None, None].expand(-1, -1, self.grid, self.grid)
        logits = self.classifier(self.context(torch.cat((sampled, context), 1)))
        return logits.float()


def load_encoder_warm_start(
    model: ToneController,
    checkpoint_path: str,
    expected_sha: str,
    expected_epoch: int = 54,
    strict_encoder: bool = False,
) -> str:
    import hashlib
    from pathlib import Path

    path = Path(checkpoint_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha:
        raise RuntimeError(f"encoder checkpoint hash mismatch {digest}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if int(state.get("epoch", -1)) != int(expected_epoch):
        raise RuntimeError(f"encoder checkpoint epoch is {state.get('epoch')}, expected {expected_epoch}")
    if "s_values" in state and tuple(state["s_values"]) != (15, 21, 31, 45, 63, 91):
        raise RuntimeError("encoder checkpoint s_values are not the Current6 bank")
    source = state["model"]
    own = model.encoder.state_dict()
    loaded = {
        k: source[k]
        for k in own
        if k in source and tuple(source[k].shape) == tuple(own[k].shape)
    }
    missing = sorted(set(own) - set(loaded))
    if strict_encoder and missing:
        raise RuntimeError(f"encoder warm start missing keys {missing}")
    model.encoder.load_state_dict({**own, **loaded}, strict=True)
    return f"loaded {len(loaded)} encoder tensors; unmatched {missing}"
