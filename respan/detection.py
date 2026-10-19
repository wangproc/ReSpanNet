"""MMDetection 2.x registration."""

from mmdet.models.builder import BACKBONES
from .backbone import ReSpanNet
from .pyramid import ContextPyramid

BACKBONES.register_module(module=ReSpanNet)

__all__ = ["ReSpanNet", "ContextPyramid"]
