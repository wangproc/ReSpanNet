"""Training recipe for maskrcnn."""

model_train_cfg = {'rpn': {'assigner': {'type': 'MaxIoUAssigner',
                      'pos_iou_thr': 0.7,
                      'neg_iou_thr': 0.3,
                      'min_pos_iou': 0.3,
                      'match_low_quality': True,
                      'ignore_iof_thr': -1},
         'sampler': {'type': 'RandomSampler',
                     'num': 256,
                     'pos_fraction': 0.5,
                     'neg_pos_ub': -1,
                     'add_gt_as_proposals': False},
         'allowed_border': -1,
         'pos_weight': -1,
         'debug': False},
 'rpn_proposal': {'nms_pre': 2000,
                  'max_per_img': 1000,
                  'nms': {'type': 'nms', 'iou_threshold': 0.7},
                  'min_bbox_size': 0},
 'rcnn': {'assigner': {'type': 'MaxIoUAssigner',
                       'pos_iou_thr': 0.5,
                       'neg_iou_thr': 0.5,
                       'min_pos_iou': 0.5,
                       'match_low_quality': True,
                       'ignore_iof_thr': -1},
          'sampler': {'type': 'RandomSampler',
                      'num': 512,
                      'pos_fraction': 0.25,
                      'neg_pos_ub': -1,
                      'add_gt_as_proposals': True},
          'mask_size': 28,
          'pos_weight': -1,
          'debug': False}}

optimizer = {'type': 'AdamW', 'lr': 0.0002, 'weight_decay': 0.0001}

optimizer_config = {'grad_clip': None}

lr_config = {'policy': 'step', 'warmup': 'linear', 'warmup_iters': 500, 'warmup_ratio': 0.001, 'step': [8, 11]}

runner = {'type': 'EpochBasedRunner', 'max_epochs': 12}

workflow = [('train', 1)]

evaluation = {'metric': ['bbox', 'segm']}

data = {'train': {'type': 'CocoDataset',
           'ann_file': 'data/coco/annotations/instances_train2017.json',
           'img_prefix': 'data/coco/train2017/',
           'pipeline': [{'type': 'LoadImageFromFile'},
                        {'type': 'LoadAnnotations', 'with_bbox': True, 'with_mask': True},
                        {'type': 'Resize', 'img_scale': (1333, 800), 'keep_ratio': True},
                        {'type': 'RandomFlip', 'flip_ratio': 0.5},
                        {'type': 'Normalize',
                         'mean': [123.675, 116.28, 103.53],
                         'std': [58.395, 57.12, 57.375],
                         'to_rgb': True},
                        {'type': 'Pad', 'size_divisor': 32},
                        {'type': 'DefaultFormatBundle'},
                        {'type': 'Collect',
                         'keys': ['img', 'gt_bboxes', 'gt_labels', 'gt_masks']}]},
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
 'samples_per_gpu': 2,
 'workers_per_gpu': 2}

checkpoint_config = {'interval': 1}

log_config = {'interval': 50, 'hooks': [{'type': 'TextLoggerHook'}]}

log_level = 'INFO'

load_from = None

resume_from = None

dist_params = {'backend': 'nccl'}
