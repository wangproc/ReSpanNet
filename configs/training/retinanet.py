"""Training recipe for retinanet."""

model_train_cfg = {'assigner': {'type': 'MaxIoUAssigner',
              'pos_iou_thr': 0.5,
              'neg_iou_thr': 0.4,
              'min_pos_iou': 0,
              'ignore_iof_thr': -1},
 'allowed_border': -1,
 'pos_weight': -1,
 'debug': False}

optimizer = {'type': 'AdamW',
 'lr': 0.0002,
 'betas': (0.9, 0.999),
 'weight_decay': 0.05,
 'paramwise_cfg': {'custom_keys': {'norm': {'decay_mult': 0.0}}}}

optimizer_config = {'grad_clip': {'max_norm': 0.1, 'norm_type': 2}}

lr_config = {'policy': 'DelayedCosineAnnealing',
 'start_epoch': 6,
 'warmup': 'linear',
 'warmup_iters': 500,
 'warmup_ratio': 1e-05,
 'min_lr': 0}

runner = {'type': 'EpochBasedRunner', 'max_epochs': 12}

workflow = [('train', 1)]

evaluation = {'interval': 1, 'metric': 'bbox'}

data = {'train': {'type': 'CocoDataset',
           'ann_file': 'data/coco/annotations/instances_train2017.json',
           'img_prefix': 'data/coco/train2017/',
           'pipeline': [{'type': 'LoadImageFromFile'},
                        {'type': 'LoadAnnotations', 'with_bbox': True},
                        {'type': 'Resize', 'img_scale': (1333, 800), 'keep_ratio': True},
                        {'type': 'RandomFlip', 'flip_ratio': 0.5},
                        {'type': 'Normalize',
                         'mean': [123.675, 116.28, 103.53],
                         'std': [58.395, 57.12, 57.375],
                         'to_rgb': True},
                        {'type': 'Pad', 'size_divisor': 32},
                        {'type': 'DefaultFormatBundle'},
                        {'type': 'Collect', 'keys': ['img', 'gt_bboxes', 'gt_labels']}]},
 'val': {'type': 'CocoDataset',
         'ann_file': 'data/coco/annotations/instances_val2017.json',
         'img_prefix': 'data/coco/val2017/',
         'pipeline': [{'type': 'LoadImageFromFile'},
                      {'type': 'MultiScaleFlipAug',
                       'img_scale': (1333, 800),
                       'flip': False,
                       'transforms': [{'type': 'Resize', 'keep_ratio': True},
                                      {'type': 'RandomFlip'},
                                      {'type': 'Normalize',
                                       'mean': [123.675, 116.28, 103.53],
                                       'std': [58.395, 57.12, 57.375],
                                       'to_rgb': True},
                                      {'type': 'Pad', 'size_divisor': 32},
                                      {'type': 'ImageToTensor', 'keys': ['img']},
                                      {'type': 'Collect', 'keys': ['img']}]}]},
 'samples_per_gpu': 4,
 'workers_per_gpu': 4}

checkpoint_config = {'interval': 1}

log_config = {'interval': 50, 'hooks': [{'type': 'TextLoggerHook'}]}

log_level = 'INFO'

load_from = None

resume_from = None

dist_params = {'backend': 'nccl'}
