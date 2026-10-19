"""Train ReSpanNet classification models with PyTorch or torchrun."""

import argparse
from contextlib import nullcontext
import math
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torchvision.datasets import ImageFolder
from timm.data import create_transform, Mixup
from timm.loss import SoftTargetCrossEntropy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from respan.backbone import ReSpanClassifier, SCALES
from respan.classification import ManifestImages, training_recipe
from respan.checkpoint import load_weights


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', type=str.lower, choices=tuple(SCALES), default='s')
    parser.add_argument('--dataset', choices=('imagenet', 'imagenet100'), default='imagenet')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--split-dir', default=str(Path(__file__).resolve().parents[1] / 'imagenet100_split'))
    parser.add_argument('--output', default='outputs/classification')
    parser.add_argument('--batch-size', type=int, default=128, help='Images per process and microbatch')
    parser.add_argument('--global-batch-size', type=int)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--image-size', type=int)
    parser.add_argument('--num-classes', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume', help='Continue from training_state.pth')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--checkpoint', help='Tensor-only classification checkpoint for evaluation')
    parser.add_argument('--local-rank', '--local_rank', type=int, default=0)
    return parser


def dataset(args, split, transform):
    if args.dataset == 'imagenet100':
        return ManifestImages(args.data_root, args.split_dir, split, transform)
    return ImageFolder(str(Path(args.data_root) / split), transform=transform)


@torch.no_grad()
def validate(model, loader, device, distributed):
    model.eval()
    totals = torch.zeros(3, dtype=torch.float64, device=device)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        totals[0] += nn.functional.cross_entropy(logits, labels, reduction='sum').double()
        totals[1] += (logits.argmax(1) == labels).sum()
        totals[2] += len(labels)
    if distributed:
        dist.all_reduce(totals)
    return (totals[0] / totals[2]).item(), (100 * totals[1] / totals[2]).item()


def learning_rate(position, recipe, peak):
    warmup = min(recipe['warmup_epochs'], recipe['epochs'])
    if position < warmup:
        return recipe['warmup_lr'] + (peak - recipe['warmup_lr']) * position / max(warmup, 1)
    progress = (position - warmup) / max(recipe['epochs'] - warmup, 1)
    return recipe['min_lr'] + 0.5 * (peak - recipe['min_lr']) * (1 + math.cos(math.pi * progress))


def run(args):
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    distributed = world > 1
    device = torch.device(args.device)
    if device.type == 'cuda':
        device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', args.local_rank)) if distributed else (device.index or 0))
        torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    recipe = training_recipe(args.variant, args.dataset)
    if args.epochs is not None:
        recipe['epochs'] = args.epochs
    classes = args.num_classes or recipe['num_classes']
    total_batch = args.global_batch_size or recipe['global_batch_size']
    if args.batch_size <= 0 or total_batch <= 0 or (not args.evaluate and total_batch % (args.batch_size * world)):
        raise ValueError('Global batch size must be divisible by batch-size times world size')
    accumulation = total_batch // (args.batch_size * world)
    if recipe['epochs'] <= 0:
        raise ValueError('Epochs must be positive')
    peak_lr = args.lr if args.lr is not None else recipe['lr'] * total_batch / recipe['global_batch_size']
    model = ReSpanClassifier(args.variant, classes).to(device)
    eval_size = args.image_size or 224
    val_transform = create_transform(input_size=eval_size, is_training=False, interpolation='bicubic', crop_pct=0.9)
    val_data = dataset(args, 'val', val_transform)
    if len(val_data.classes) != classes:
        raise ValueError('Validation classes do not match num-classes')
    val_part = Subset(val_data, range(rank, len(val_data), world)) if distributed else val_data
    val_loader = DataLoader(val_part, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=device.type == 'cuda')
    if args.evaluate:
        if not args.checkpoint:
            raise ValueError('--evaluate requires --checkpoint')
        load_weights(model, args.checkpoint)
        loss, accuracy = validate(model, val_loader, device, distributed)
        if rank == 0:
            print(f'Validation loss: {loss:.4f}; Top-1: {accuracy:.3f}')
        if distributed:
            dist.destroy_process_group()
        return
    train_data = dataset(args, 'train', None)
    if train_data.class_to_idx != val_data.class_to_idx:
        raise ValueError('Training and validation label mappings differ')
    sampler = DistributedSampler(train_data, num_replicas=world, rank=rank, shuffle=True, seed=args.seed) if distributed else None
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (no_decay if parameter.ndim <= 1 or name.endswith('.bias') else decay).append(parameter)
    optimizer = torch.optim.AdamW([{'params': decay}, {'params': no_decay, 'weight_decay': 0.0}],
                                 lr=peak_lr, weight_decay=recipe['weight_decay'], betas=(0.9, 0.999))
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')
    start, best = 0, float('-inf')
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        if checkpoint['variant'] != args.variant or checkpoint['dataset'] != args.dataset:
            raise ValueError('Resume checkpoint uses a different variant or dataset')
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        start, best = checkpoint['epoch'], checkpoint['best_accuracy']
    wrapped = nn.parallel.DistributedDataParallel(model, device_ids=[device.index] if device.type == 'cuda' else None,
                                                  broadcast_buffers=False) if distributed else model
    mix = Mixup(mixup_alpha=recipe['mixup'], cutmix_alpha=1.0, prob=1.0, switch_prob=0.5,
                mode='batch', label_smoothing=0.1, num_classes=classes)
    criterion = SoftTargetCrossEntropy()
    output = Path(args.output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    for epoch in range(start, recipe['epochs']):
        size = args.image_size or (192 if epoch < recipe['resize_epoch'] else 224)
        train_data.transform = create_transform(input_size=size, is_training=True, color_jitter=0,
                                                auto_augment=recipe['auto_augment'], interpolation='bicubic', re_prob=0)
        if sampler is not None:
            sampler.set_epoch(epoch)
        loader = DataLoader(train_data, batch_size=args.batch_size, sampler=sampler, shuffle=sampler is None,
                            num_workers=args.workers, pin_memory=device.type == 'cuda', drop_last=True)
        if len(loader) == 0 or args.batch_size % 2:
            raise ValueError('Training requires an even microbatch size and at least one full batch')
        wrapped.train()
        optimizer.zero_grad(set_to_none=True)
        totals = torch.zeros(2, dtype=torch.float64, device=device)
        for step, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            images, labels = mix(images, labels)
            group_start = (step // accumulation) * accumulation
            group_size = min(accumulation, len(loader) - group_start)
            update = (step - group_start + 1) == group_size
            context = wrapped.no_sync() if distributed and not update else nullcontext()
            with context:
                with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                    loss = criterion(wrapped(images), labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite classification loss')
                scaler.scale(loss / group_size).backward()
            totals[0] += loss.detach().double() * len(images)
            totals[1] += len(images)
            if update:
                lr = learning_rate(epoch + group_start / len(loader), recipe, peak_lr)
                for group in optimizer.param_groups:
                    group['lr'] = lr
                if recipe['clip_grad'] is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), recipe['clip_grad'])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        if distributed:
            dist.all_reduce(totals)
            for buffer in model.buffers():
                dist.broadcast(buffer, src=0)
        val_loss, accuracy = validate(model, val_loader, device, distributed)
        if rank == 0:
            print(f'Epoch {epoch + 1}/{recipe["epochs"]}: train loss {totals[0]/totals[1]:.4f}, '
                  f'validation loss {val_loss:.4f}, Top-1 {accuracy:.3f}', flush=True)
            state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            torch.save(state, output / 'model_last.pth')
            if accuracy > best:
                best = accuracy
                torch.save(state, output / 'model_best.pth')
            torch.save(dict(model=state, optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                            epoch=epoch + 1, best_accuracy=best, variant=args.variant, dataset=args.dataset),
                       output / 'training_state.pth')
        if distributed:
            dist.barrier()
    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    run(build_parser().parse_args())
