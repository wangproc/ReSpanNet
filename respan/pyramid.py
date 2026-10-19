"""Two-input feature pyramid used by the RetinaNet configuration.

Adapted from the EfficientViT/MobileMamba downstream FPN implementation.
Copyright (c) 2022 Microsoft. See THIRD_PARTY_NOTICES.md and LICENSES/.
"""

from torch import nn
from torch.nn import functional as F
from mmcv.cnn import ConvModule
from mmdet.models.builder import NECKS


@NECKS.register_module()
class ContextPyramid(nn.Module):
    """Produce strides 8--128 from stride-16/32 backbone features."""

    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        if len(in_channels) != 2:
            raise ValueError("ContextPyramid requires two backbone feature levels")
        self.lateral_convs = nn.ModuleList([
            ConvModule(c, out_channels, 1, act_cfg=None, inplace=False)
            for c in in_channels
        ])
        self.fpn_convs = nn.ModuleList([
            ConvModule(out_channels, out_channels, 3, padding=1, act_cfg=None, inplace=False)
            for _ in in_channels
        ])
        upsample = ConvModule(out_channels, out_channels, 2, stride=2, act_cfg=None, inplace=False)
        upsample.conv = nn.ConvTranspose2d(out_channels, out_channels, 2, stride=2)
        self.extra_trans_convs = nn.ModuleList([upsample])
        self.extra_fpn_convs = nn.ModuleList([
            ConvModule(out_channels, out_channels, 3, padding=1, act_cfg=None, inplace=False)
        ])

    def init_weights(self):
        pass

    def forward(self, inputs):
        fine, coarse = [layer(x) for layer, x in zip(self.lateral_convs, inputs)]
        fine = fine + F.interpolate(coarse, size=fine.shape[-2:], mode="nearest")
        p3 = self.extra_fpn_convs[0](self.extra_trans_convs[0](fine))
        p4, p5 = [layer(x) for layer, x in zip(self.fpn_convs, (fine, coarse))]
        p6 = F.max_pool2d(p5, 1, stride=2)
        p7 = F.max_pool2d(p6, 1, stride=2)
        return p3, p4, p5, p6, p7
