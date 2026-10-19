"""Evaluate a task checkpoint on COCO val2017 or ADE20K validation."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from respan.checkpoint import load_weights
from respan.task_config import task_config, MODELS


def evaluate_detection(args):
    from mmcv import Config
    import torch
    from mmcv.parallel import MMDataParallel, scatter
    from mmdet.apis import single_gpu_test
    from mmdet.datasets import build_dataset, build_dataloader
    from mmdet.models import build_detector
    import respan.detection

    cfg = Config.fromfile(args.config) if args.config else Config(task_config(args.model, args.variant))
    dataset_cfg = cfg.data.test.copy()
    if args.data_root:
        root = Path(args.data_root)
        dataset_cfg.ann_file = str(root / "annotations/instances_val2017.json")
        dataset_cfg.img_prefix = str(root / "val2017") + "/"
    dataset_cfg.test_mode = True
    dataset = build_dataset(dataset_cfg)
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=args.workers,
                              dist=False, shuffle=False)
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.get("test_cfg"))
    load_weights(model, args.checkpoint)
    model.CLASSES = dataset.CLASSES
    model.eval()
    if args.device == "cpu":
        from mmdet.core import encode_mask_results
        predictions = []
        with torch.no_grad():
            for batch in loader:
                inputs = scatter(batch, [-1])[0]
                result = model(return_loss=False, rescale=True, **inputs)
                if isinstance(result[0], tuple):
                    result = [(boxes, encode_mask_results(masks)) for boxes, masks in result]
                predictions.extend(result)
    else:
        device = torch.device(args.device)
        if device.type != "cuda":
            raise ValueError("Detection supports cpu or cuda devices")
        device_ids = [device.index if device.index is not None else 0]
        model = model.to(args.device)
        model = MMDataParallel(model, device_ids=device_ids)
        predictions = single_gpu_test(model, loader, show=False)
    metrics = ["bbox", "segm"] if cfg.model.type == "MaskRCNN" else ["bbox"]
    return dataset.evaluate(predictions, metric=metrics)


def evaluate_segmentation(args):
    import torch
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmengine.model import revert_sync_batchnorm
    from mmseg.registry import MODELS, METRICS
    from mmseg.utils import register_all_modules
    import respan.segmentation

    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(args.config) if args.config else Config(task_config(args.model, args.variant))
    loader_cfg = cfg.test_dataloader.copy()
    if args.data_root:
        loader_cfg.dataset.data_root = args.data_root
    loader_cfg.num_workers = args.workers
    loader_cfg.persistent_workers = args.workers > 0
    loader = Runner.build_dataloader(loader_cfg)
    model = MODELS.build(cfg.model)
    load_weights(model, args.checkpoint)
    if args.device == "cpu":
        model = revert_sync_batchnorm(model)
    model.to(args.device).eval()
    metric = METRICS.build(dict(type="IoUMetric", iou_metrics=["mIoU"]))
    metric.dataset_meta = loader.dataset.metainfo
    model.dataset_meta = loader.dataset.metainfo
    with torch.no_grad():
        for batch in loader:
            predictions = [sample.to_dict() for sample in model.test_step(batch)]
            metric.process(batch, predictions)
    return metric.evaluate(len(loader.dataset))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("detection", "segmentation"))
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--config")
    selection.add_argument("--model", choices=tuple(MODELS))
    parser.add_argument("--variant", type=str.lower, choices=('s', 'm', 'l'), default='s')
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.model:
        inferred = MODELS[args.model][1]
        if args.task and args.task != inferred:
            parser.error('--task does not match --model')
        args.task = inferred
    elif not args.task:
        parser.error('--config requires --task')
    fn = evaluate_detection if args.task == "detection" else evaluate_segmentation
    print(fn(args))


if __name__ == "__main__":
    main()
