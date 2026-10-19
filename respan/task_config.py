"""Select detector and segmentation configurations by model and backbone scale."""

import copy
from pathlib import Path
import runpy

from .backbone import SCALES

MODELS = {'maskrcnn': ('mask_rcnn_respan_s.py', 'detection'),
          'retinanet': ('retinanet_respan_s.py', 'detection'),
          'deeplabv3': ('deeplabv3_respan_s.py', 'segmentation'),
          'pspnet': ('pspnet_respan_s.py', 'segmentation')}


def task_config(name, variant='s', training=False):
    variant = variant.lower()
    if variant not in ('s', 'm', 'l'):
        raise ValueError('Task backbones use S, M or L')
    root = Path(__file__).resolve().parents[1] / 'configs'
    filename, task = MODELS[name]
    cfg = {key: copy.deepcopy(value) for key, value in runpy.run_path(str(root / filename)).items()
           if not key.startswith('__')}
    width = SCALES[variant][0]
    cfg['model']['backbone']['variant'] = variant
    if name == 'maskrcnn':
        cfg['model']['neck']['in_channels'] = [width * 2 ** i for i in range(4)]
    elif name == 'retinanet':
        cfg['model']['neck']['in_channels'] = [4 * width, 8 * width]
    else:
        cfg['model']['decode_head']['in_channels'] = 8 * width
        cfg['model']['auxiliary_head']['in_channels'] = 4 * width
    if training:
        if task == 'segmentation':
            cfg['test_cfg'] = dict(type='TestLoop')
        training_cfg = runpy.run_path(str(root / 'training' / (name + '.py')))
        cfg['model']['train_cfg'] = copy.deepcopy(training_cfg['model_train_cfg'])
        for key, value in training_cfg.items():
            if key != 'model_train_cfg' and not key.startswith('__'):
                if key == 'data':
                    cfg['data'].update(copy.deepcopy(value))
                else:
                    cfg[key] = copy.deepcopy(value)
    return cfg
