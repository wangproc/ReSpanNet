"""Inference and validation configuration."""

custom_imports = {'allow_failed_imports': False, 'imports': ['respan.segmentation']}

model = {'auxiliary_head': {'align_corners': False,
                    'channels': 256,
                    'concat_input': False,
                    'dropout_ratio': 0.1,
                    'in_channels': 416,
                    'in_index': 0,
                    'loss_decode': {'loss_weight': 0.4,
                                    'type': 'CrossEntropyLoss',
                                    'use_sigmoid': False},
                    'norm_cfg': {'requires_grad': True, 'type': 'SyncBN'},
                    'num_classes': 150,
                    'num_convs': 1,
                    'type': 'FCNHead'},
 'backbone': {'norm_eval': False, 'out_indices': (2, 3), 'type': 'ReSpanNet', 'variant': 's'},
 'data_preprocessor': {'bgr_to_rgb': True,
                       'mean': [123.675, 116.28, 103.53],
                       'pad_val': 0,
                       'seg_pad_val': 255,
                       'size': (512, 512),
                       'std': [58.395, 57.12, 57.375],
                       'type': 'SegDataPreProcessor'},
 'decode_head': {'align_corners': False,
                 'channels': 256,
                 'dilations': (1, 12, 24, 36),
                 'dropout_ratio': 0.1,
                 'in_channels': 832,
                 'in_index': 1,
                 'loss_decode': {'loss_weight': 1.0,
                                 'type': 'CrossEntropyLoss',
                                 'use_sigmoid': False},
                 'norm_cfg': {'requires_grad': True, 'type': 'SyncBN'},
                 'num_classes': 150,
                 'type': 'ASPPHead'},
 'test_cfg': {'mode': 'whole'},
 'type': 'EncoderDecoder'}

default_scope = 'mmseg'

test_dataloader = {'batch_size': 1,
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

test_pipeline = [{'type': 'LoadImageFromFile'},
 {'keep_ratio': True, 'scale': (2048, 512), 'type': 'Resize'},
 {'reduce_zero_label': True, 'type': 'LoadAnnotations'},
 {'type': 'PackSegInputs'}]

test_evaluator = {'iou_metrics': ['mIoU'], 'type': 'IoUMetric'}
