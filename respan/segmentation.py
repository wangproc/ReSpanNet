"""MMSegmentation 1.x registration."""

from mmseg.registry import MODELS
from .backbone import ReSpanNet

MODELS.register_module(module=ReSpanNet)
