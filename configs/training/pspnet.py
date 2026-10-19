"""Training recipe for pspnet."""

model_train_cfg = {}

train_cfg = {'max_iters': 80000, 'type': 'IterBasedTrainLoop', 'val_interval': 8000}

train_dataloader = {'batch_size': 8,
 'dataset': {'data_prefix': {'img_path': 'images/training', 'seg_map_path': 'annotations/training'},
             'data_root': 'data/ade/ADEChallengeData2016',
             'pipeline': [{'type': 'LoadImageFromFile'},
                          {'reduce_zero_label': True, 'type': 'LoadAnnotations'},
                          {'keep_ratio': True,
                           'ratio_range': (0.5, 2.0),
                           'scale': (2048, 512),
                           'type': 'RandomResize'},
                          {'cat_max_ratio': 0.75, 'crop_size': (512, 512), 'type': 'RandomCrop'},
                          {'prob': 0.5, 'type': 'RandomFlip'},
                          {'type': 'PhotoMetricDistortion'},
                          {'type': 'PackSegInputs'}],
             'type': 'ADE20KDataset'},
 'num_workers': 8,
 'persistent_workers': True,
 'sampler': {'shuffle': True, 'type': 'InfiniteSampler'}}

optim_wrapper = {'clip_grad': {'max_norm': 0.1, 'norm_type': 2},
 'optimizer': {'betas': (0.9, 0.999), 'lr': 0.00012, 'type': 'AdamW', 'weight_decay': 0.05},
 'paramwise_cfg': {'custom_keys': {'absolute_pos_embed': {'decay_mult': 0.0},
                                   'norm': {'decay_mult': 0.0},
                                   'relative_position_bias_table': {'decay_mult': 0.0}}},
 'type': 'OptimWrapper'}

param_scheduler = [{'begin': 0, 'by_epoch': False, 'end': 500, 'start_factor': 1e-05, 'type': 'LinearLR'},
 {'T_max': 40000,
  'begin': 40000,
  'by_epoch': False,
  'end': 80000,
  'eta_min': 0,
  'type': 'CosineAnnealingLR'}]

val_cfg = {'type': 'ValLoop'}

val_dataloader = {'batch_size': 1,
 'dataset': {'data_prefix': {'img_path': 'images/validation',
                             'seg_map_path': 'annotations/validation'},
             'data_root': 'data/ade/ADEChallengeData2016',
             'pipeline': [{'type': 'LoadImageFromFile'},
                          {'keep_ratio': True, 'scale': (2048, 512), 'type': 'Resize'},
                          {'reduce_zero_label': True, 'type': 'LoadAnnotations'},
                          {'type': 'PackSegInputs'}],
             'type': 'ADE20KDataset'},
 'num_workers': 2,
 'persistent_workers': True,
 'sampler': {'shuffle': False, 'type': 'DefaultSampler'}}

val_evaluator = {'iou_metrics': ['mIoU'], 'type': 'IoUMetric'}

default_hooks = {'timer': {'type': 'IterTimerHook'},
 'logger': {'type': 'LoggerHook', 'interval': 50},
 'param_scheduler': {'type': 'ParamSchedulerHook'},
 'checkpoint': {'type': 'CheckpointHook', 'interval': 8000, 'by_epoch': False},
 'sampler_seed': {'type': 'DistSamplerSeedHook'}}

env_cfg = {'cudnn_benchmark': True,
 'mp_cfg': {'mp_start_method': 'spawn', 'opencv_num_threads': 0},
 'dist_cfg': {'backend': 'nccl'}}

log_processor = {'by_epoch': False}
