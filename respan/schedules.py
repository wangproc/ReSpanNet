"""Learning-rate schedules for detector training."""

import math
from mmcv.runner.hooks import HOOKS
from mmcv.runner.hooks.lr_updater import LrUpdaterHook


@HOOKS.register_module()
class DelayedCosineAnnealingLrUpdaterHook(LrUpdaterHook):
    def __init__(self, start_epoch, min_lr=0, **kwargs):
        self.start_epoch = start_epoch
        self.min_lr = min_lr
        super().__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        start = self.start_epoch if self.by_epoch else self.start_epoch * len(runner.data_loader)
        current = runner.epoch if self.by_epoch else runner.iter
        end = runner.max_epochs if self.by_epoch else runner.max_iters
        fraction = max(0.0, (current - start) / max(end - start, 1))
        return self.min_lr + (base_lr - self.min_lr) * (1 + math.cos(math.pi * fraction)) / 2
