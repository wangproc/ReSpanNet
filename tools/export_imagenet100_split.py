"""Export ImageNet-100 class labels and image-level partition manifests."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'}


def export(root, output):
    root, output = Path(root), Path(output)
    classes = sorted(path.name for path in (root / 'train').iterdir() if path.is_dir())
    if len(classes) != 100:
        raise ValueError('Expected 100 class directories')
    output.mkdir(parents=True, exist_ok=True)
    counts = {name: {} for name in classes}
    identifiers = {name: Counter() for name in classes}
    nonstandard = Counter()
    names, totals, sizes = {}, {}, {}
    for split in ('train', 'val', 'test'):
        if sorted(path.name for path in (root / split).iterdir() if path.is_dir()) != classes:
            raise ValueError('Class directories differ across partitions')
        rows, filenames, size = [], set(), 0
        for index, name in enumerate(classes):
            files = sorted(path for path in (root / split / name).rglob('*')
                           if path.is_file() and path.suffix.lower() in EXTENSIONS)
            counts[name][split] = len(files)
            for path in files:
                rows.append('{} {}\n'.format(path.relative_to(root).as_posix(), index))
                filenames.add(path.name)
                size += path.stat().st_size
                match = re.match(r'(n\d{8})_', path.name)
                if match:
                    identifiers[name][match.group(1)] += 1
                else:
                    nonstandard[name] += 1
        (output / (split + '.txt')).write_bytes(''.join(rows).encode('utf-8'))
        names[split], totals[split], sizes[split] = filenames, len(rows), size
    with (output / 'classes.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['class_index', 'class_name', 'train_count', 'val_count', 'test_count',
                         'observed_filename_synsets', 'nonstandard_filename_count'])
        for index, name in enumerate(classes):
            writer.writerow([index, name, counts[name]['train'], counts[name]['val'], counts[name]['test'],
                             ';'.join(sorted(identifiers[name])), nonstandard[name]])
    summary = dict(class_count=len(classes), image_counts=totals, image_bytes=sizes,
        class_index_rule='Lexicographic directory order, as in ImageFolder',
        manifest_format='dataset-root-relative POSIX path, space, integer class index',
        cross_split_filename_overlap={a+'_'+b: len(names[a] & names[b]) for a,b in
                                      [('train','val'),('train','test'),('val','test')]},
        export_command='python tools/export_imagenet100_split.py --root data/ImageNet100 --output imagenet100_split',
        evaluation_partition='val', selection_partition='val',
        classes_with_multiple_filename_synsets={name: dict(ids) for name,ids in identifiers.items() if len(ids)>1},
        classes_with_nonstandard_filenames=dict(nonstandard),
        manifest_sha256={name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                         for name in ('classes.csv','train.txt','val.txt','test.txt')})
    (output / 'split_summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    return summary


def verify(root, split_dir):
    root, split_dir = Path(root), Path(split_dir)
    summary = json.loads((split_dir / 'split_summary.json').read_text(encoding='utf-8'))
    for name, digest in summary['manifest_sha256'].items():
        if hashlib.sha256((split_dir / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Manifest checksum mismatch: ' + name)
    with (split_dir / 'classes.csv').open(encoding='utf-8', newline='') as handle:
        classes = list(csv.DictReader(handle))
    for split in ('train', 'val', 'test'):
        counts = Counter()
        for line in (split_dir / (split+'.txt')).read_text(encoding='utf-8').splitlines():
            relative, label = line.rsplit(' ', 1)
            label = int(label)
            parts = relative.split('/')
            if not 0 <= label < len(classes) or '..' in parts or ':' in relative or '\\' in relative:
                raise ValueError('Invalid manifest record')
            if parts[:2] != [split, classes[label]['class_name']]:
                raise ValueError('Manifest label and directory disagree')
            if not root.joinpath(*parts).is_file():
                raise FileNotFoundError(relative)
            counts[label] += 1
        if sum(counts.values()) != summary['image_counts'][split]:
            raise ValueError('Partition count mismatch')
        for label, row in enumerate(classes):
            if counts[label] != int(row[split+'_count']):
                raise ValueError('Class count mismatch: ' + row['class_name'])
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', default='imagenet100_split')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    summary = (verify if args.verify else export)(args.root, args.output)
    print(json.dumps({'class_count': summary['class_count'], 'image_counts': summary['image_counts']}))


if __name__ == '__main__':
    main()
