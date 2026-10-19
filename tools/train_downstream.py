"""Train a detector or segmentation model with a selected ReSpanNet scale."""

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from respan.task_config import task_config, MODELS


def detection(args):
    import torch
    from mmcv import Config
    from mmcv.runner import init_dist
    from mmdet.apis import train_detector, set_random_seed
    from mmdet.datasets import build_dataset
    from mmdet.models import build_detector
    import respan.detection
    import respan.schedules

    cfg = Config(task_config(args.model, args.variant, training=True))
    distributed = int(os.environ.get('WORLD_SIZE', '1')) > 1
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if distributed:
        init_dist('pytorch', backend='nccl' if cfg.device == 'cuda' else 'gloo')
    cfg.gpu_ids = list(range(int(os.environ.get('WORLD_SIZE', '1')))) if distributed else [0]
    cfg.work_dir = args.output
    cfg.seed = args.seed
    set_random_seed(args.seed, deterministic=False)
    root = Path(args.data_root)
    for split in ('train', 'val', 'test'):
        part = 'train2017' if split == 'train' else 'val2017'
        cfg.data[split].ann_file = str(root / 'annotations' / ('instances_' + part + '.json'))
        cfg.data[split].img_prefix = str(root / part) + '/'
    cfg.data.workers_per_gpu = args.workers
    cfg.data.samples_per_gpu = args.batch_size or 2
    cfg.optimizer.lr *= cfg.data.samples_per_gpu * len(cfg.gpu_ids) / 16
    if args.max_steps:
        cfg.runner = dict(type='IterBasedRunner', max_iters=args.max_steps)
        cfg.lr_config.by_epoch = False
        cfg.checkpoint_config = dict(interval=args.max_steps, by_epoch=False)
    if args.backbone_checkpoint:
        cfg.model.backbone.init_cfg = dict(type='Pretrained', checkpoint=args.backbone_checkpoint)
    model = build_detector(cfg.model)
    model.init_weights()
    dataset = build_dataset(cfg.data.train)
    model.CLASSES = dataset.CLASSES
    train_detector(model, [dataset], cfg, distributed=distributed, validate=not args.no_validate)


def segmentation(args):
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmseg.utils import register_all_modules
    import respan.segmentation

    register_all_modules(init_default_scope=True)
    cfg = Config(task_config(args.model, args.variant, training=True))
    cfg.work_dir = args.output
    cfg.launcher = 'pytorch' if int(os.environ.get('WORLD_SIZE', '1')) > 1 else 'none'
    cfg.randomness = dict(seed=args.seed)
    for split in ('train', 'val', 'test'):
        loader = cfg[split + '_dataloader']
        loader.dataset.data_root = args.data_root
        loader.num_workers = args.workers
        loader.persistent_workers = args.workers > 0
    cfg.train_dataloader.batch_size = args.batch_size or cfg.train_dataloader.batch_size
    if args.max_steps:
        cfg.train_cfg.max_iters = args.max_steps
    if args.no_validate:
        cfg.val_cfg = cfg.val_dataloader = cfg.val_evaluator = None
    if args.backbone_checkpoint:
        cfg.model.backbone.init_cfg = dict(type='Pretrained', checkpoint=args.backbone_checkpoint)
    Runner.from_cfg(cfg).train()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=tuple(MODELS), required=True)
    parser.add_argument('--variant', type=str.lower, choices=('s', 'm', 'l'), default='s')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', default='outputs/task_training')
    parser.add_argument('--backbone-checkpoint')
    parser.add_argument('--batch-size', type=int, help='Images per GPU')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--no-validate', action='store_true')
    parser.add_argument('--local-rank', '--local_rank', type=int, default=0)
    args = parser.parse_args()
    os.environ.setdefault('LOCAL_RANK', str(args.local_rank))
    Path(args.output).mkdir(parents=True, exist_ok=True)
    globals()[MODELS[args.model][1]](args)


if __name__ == '__main__':
    main()
