"""Spatial refinement within stages and context enrichment at transitions.

See THIRD_PARTY_NOTICES.md for implementation acknowledgments.
"""

import torch
from torch import nn
from torch.nn import functional as F


SCALES = {
    "t0": (40, (1, 2, 8, 2), "gelu", 0.0, 0.0),
    "t1": (56, (1, 2, 9, 2), "gelu", 0.02, 0.0),
    "t2": (80, (2, 2, 9, 3), "relu", 0.05, 0.05),
    "s": (104, (2, 2, 14, 3), "relu", 0.10, 0.08),
    "m": (120, (3, 4, 19, 4), "relu", 0.20, 0.08),
    "l": (160, (3, 4, 21, 4), "relu", 0.30, 0.08),
}


class ResponseModulation(nn.Module):
    """A spatially shared response coefficient for each channel."""

    def __init__(self, channels, epsilon=0.1, noise=0.0):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(channels, 1, 1))
        self.offset = nn.Parameter(torch.full((channels, 1, 1), -1.0))
        self.epsilon = epsilon
        self.noise = noise

    def forward(self, x):
        logits = self.scale * F.adaptive_avg_pool2d(x, 1) + self.offset
        if self.training and self.noise:
            logits = logits + torch.randn_like(logits) * self.noise
        return self.epsilon + (1 - 2 * self.epsilon) * logits.sigmoid()


class CSRConv(nn.Module):
    """Complementary Spatial Refinement Convolution."""

    def __init__(self, channels, noise=0.0):
        super().__init__()
        self.core_width = channels // 4
        self.complement_width = channels - self.core_width
        self.core = nn.Conv2d(self.core_width, self.core_width, 3, padding=1, bias=False)
        self.refine = nn.Conv2d(
            self.complement_width, self.complement_width, 3, padding=1,
            groups=self.complement_width, bias=False,
        )
        self.response = ResponseModulation(self.complement_width, noise=noise)

    def forward(self, x):
        core, complement = x.split((self.core_width, self.complement_width), dim=1)
        coefficient = self.response(complement)
        refined = coefficient * self.refine(complement) + (1 - coefficient) * complement
        return torch.cat((self.core(core), refined), dim=1)


class CED(nn.Module):
    """Context-Enriched Downsampling with a retained regional grid."""

    def __init__(self, channels, noise=0.0):
        super().__init__()
        self.response = ResponseModulation(channels, noise=noise)
        self.project = nn.Conv2d(channels, channels, 1, bias=False)
        self.reduce = nn.Conv2d(channels, 2 * channels, 2, stride=2, bias=False)
        self.norm = nn.BatchNorm2d(2 * channels)

    def forward(self, x):
        regional = F.adaptive_avg_pool2d(x, 7)
        global_context = F.adaptive_avg_pool2d(x, 1).expand_as(regional)
        coefficient = self.response(x)
        context = coefficient * global_context + (1 - coefficient) * regional
        context = F.interpolate(context, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.norm(self.reduce(x + self.project(context)))


class ResidualBlock(nn.Module):
    def __init__(self, channels, activation, drop_probability, noise):
        super().__init__()
        self.spatial = CSRConv(channels, noise)
        self.channel = nn.Sequential(
            nn.Conv2d(channels, 2 * channels, 1, bias=False),
            nn.BatchNorm2d(2 * channels),
            nn.GELU() if activation == "gelu" else nn.ReLU(inplace=True),
            nn.Conv2d(2 * channels, channels, 1, bias=False),
        )
        self.drop_probability = drop_probability

    def forward(self, x):
        update = self.channel(self.spatial(x))
        if self.training and self.drop_probability:
            keep = 1 - self.drop_probability
            mask = update.new_empty((len(x), 1, 1, 1)).bernoulli_(keep)
            update = update * mask / keep
        return x + update


class ReSpanNet(nn.Module):
    """Four-stage feature backbone; ``out_indices`` uses stage indices 0--3."""

    def __init__(self, variant="s", out_indices=(0, 1, 2, 3), norm_eval=False, init_cfg=None):
        super().__init__()
        if variant not in SCALES:
            raise ValueError("Unknown ReSpanNet variant: " + variant)
        self.out_indices = tuple(out_indices)
        if not self.out_indices or tuple(sorted(set(self.out_indices))) != self.out_indices:
            raise ValueError("out_indices must be nonempty, unique and increasing")
        if not set(self.out_indices) <= {0, 1, 2, 3}:
            raise ValueError("out_indices must refer to stages 0--3")
        width, depths, activation, max_drop, noise = SCALES[variant]
        self.norm_eval = norm_eval
        self.init_cfg = init_cfg
        self.stem = nn.Sequential(nn.Conv2d(3, width, 4, stride=4, bias=False), nn.BatchNorm2d(width))
        self.stages = nn.ModuleList()
        self.transitions = nn.ModuleList()
        self.output_norms = nn.ModuleDict()
        probabilities = torch.linspace(0, max_drop, sum(depths)).tolist()
        offset = 0
        for stage, depth in enumerate(depths):
            channels = width * 2 ** stage
            self.stages.append(nn.Sequential(*[
                ResidualBlock(channels, activation, probabilities[offset + j], noise)
                for j in range(depth)
            ]))
            if stage < 3:
                self.transitions.append(CED(channels, noise))
            if stage in self.out_indices:
                self.output_norms[str(stage)] = nn.BatchNorm2d(channels)
            offset += depth

    def init_weights(self):
        """Initialize the backbone, optionally from a classification checkpoint."""
        self.apply(initialize_parameters)
        if self.init_cfg is not None:
            from .checkpoint import initialize_backbone
            initialize_backbone(self, self.init_cfg['checkpoint'])

    def forward_features(self, x):
        x = self.stem(x)
        for stage, blocks in enumerate(self.stages):
            x = blocks(x)
            if stage < len(self.transitions):
                x = self.transitions[stage](x)
        return x

    def forward(self, x):
        x = self.stem(x)
        outputs = []
        for stage, blocks in enumerate(self.stages):
            x = blocks(x)
            if stage in self.out_indices:
                outputs.append(self.output_norms[str(stage)](x))
            if stage < 3:
                x = self.transitions[stage](x)
        return tuple(outputs)

    def train(self, mode=True):
        super().train(mode)
        if mode and self.norm_eval:
            for layer in self.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.eval()
        return self


def initialize_parameters(module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ReSpanClassifier(nn.Module):
    """ReSpanNet with global pooling, a projection and a classification head."""

    def __init__(self, variant="s", num_classes=1000):
        super().__init__()
        self.backbone = ReSpanNet(variant)
        self.backbone.output_norms = nn.ModuleDict()
        width, _, activation, _, _ = SCALES[variant]
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Conv2d(8 * width, 1280, 1, bias=False)
        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU(inplace=True)
        self.head = nn.Linear(1280, num_classes)
        self.apply(initialize_parameters)

    def forward(self, x):
        x = self.pool(self.backbone.forward_features(x))
        return self.head(self.activation(self.project(x)).flatten(1))
