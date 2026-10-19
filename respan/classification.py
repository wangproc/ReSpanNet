"""Classification recipes and manifest-backed image loading."""

import csv
import json
from pathlib import Path, PurePosixPath

from PIL import Image
from torch.utils.data import Dataset


def training_recipe(variant, dataset):
    config = Path(__file__).resolve().parents[1] / 'configs/classification' / (dataset + '.json')
    presets = json.loads(config.read_text(encoding='utf-8'))
    if variant not in presets['variants']:
        raise ValueError('No training recipe for {} on {}'.format(variant, dataset))
    recipe = dict(presets['common'])
    recipe.update(presets['variants'][variant])
    return recipe


class ManifestImages(Dataset):
    """Read the exact labels and relative image paths recorded in a split."""

    def __init__(self, root, split_dir, split, transform=None):
        self.root = Path(root)
        split_dir = Path(split_dir)
        with (split_dir / 'classes.csv').open(encoding='utf-8', newline='') as handle:
            records = list(csv.DictReader(handle))
        if [int(row['class_index']) for row in records] != list(range(len(records))):
            raise ValueError('Class indices must be consecutive from zero')
        self.classes = [row['class_name'] for row in records]
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        self.samples = []
        for line in (split_dir / (split + '.txt')).read_text(encoding='utf-8').splitlines():
            relative, label = line.rsplit(' ', 1)
            path, label = PurePosixPath(relative), int(label)
            if path.is_absolute() or '..' in path.parts or '\\' in relative or ':' in relative:
                raise ValueError('Manifest paths must be dataset-relative POSIX paths')
            if not 0 <= label < len(self.classes) or path.parts[:2] != (split, self.classes[label]):
                raise ValueError('Manifest path and class label disagree')
            self.samples.append((self.root.joinpath(*path.parts), label))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as source:
            image = source.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, label
