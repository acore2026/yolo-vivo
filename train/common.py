"""No training-framework imports: safe validation utilities."""
from pathlib import Path
import hashlib
import json
import math
import yaml

ROOT = Path(__file__).resolve().parents[1]

def resolve(value):
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def config(path):
    return yaml.safe_load(resolve(path).read_text())

def validate_manifest(path, partition_policy='capture_group'):
    path = resolve(path)
    rows = json.loads(path.read_text())
    if partition_policy not in ('capture_group', 'same_setup_video_holdout'):
        raise ValueError('Unknown partition policy')
    counts, groups, keys = {}, {}, set()
    video_splits, image_splits = {}, {}
    for r in rows:
        if r['status'] != 'accepted':
            continue
        image = path.parent / r['image']
        if sha(image) != r['image_sha256']:
            raise ValueError(f'Image hash mismatch: {image}')
        key = (r['source_sha256'], r['frame'])
        if key in keys:
            raise ValueError(f'Duplicate frame: {key}')
        keys.add(key)
        split = r['split']
        counts[split] = counts.get(split, 0) + 1
        groups.setdefault(r['group'], set()).add(split)
        video_splits.setdefault(r['source_sha256'], set()).add(split)
        image_splits.setdefault(r['image_sha256'], set()).add(split)
        if not r['objects']:
            raise ValueError(f'Accepted empty annotation: {r["id"]}')
        for o in r['objects']:
            x1,y1,x2,y2 = o['bbox_xyxy']
            if o['class_id'] not in range(4) or not all(math.isfinite(v) for v in (x1,y1,x2,y2)) or not (0<=x1<x2<=r['width'] and 0<=y1<y2<=r['height']):
                raise ValueError(f'Invalid annotation: {r["id"]}')
    if any(len(s)>1 for s in video_splits.values()) or any(len(s)>1 for s in image_splits.values()):
        raise ValueError('Source video or identical image leaks across splits')
    if partition_policy == 'capture_group' and any(len(s)>1 for s in groups.values()):
        raise ValueError('Capture group leaks across splits')
    return rows, counts, groups

def validate_training_labels(rows, root):
    """Check actual on-disk labels against reviewed coordinates before training."""
    expected_images, expected_labels = set(), set()
    for r in rows:
        if r['status'] != 'accepted':
            continue
        image = (root / r['image']).resolve()
        expected = (root / 'images' / r['split'] / (r['id'] + '.png')).resolve()
        if image != expected:
            raise ValueError(f'Image outside prepared split: {image}')
        label = root / 'labels' / r['split'] / (r['id'] + '.txt')
        lines = label.read_text().splitlines()
        if len(lines) != len(r['objects']):
            raise ValueError(f'Object count mismatch: {label}')
        for line, obj in zip(lines, r['objects']):
            parts = line.split()
            if len(parts) != 5 or int(parts[0]) != obj['class_id']:
                raise ValueError(f'Class/format mismatch: {label}')
            x,y,w,h = map(float, parts[1:])
            box = [(x-w/2)*r['width'], (y-h/2)*r['height'],
                   (x+w/2)*r['width'], (y+h/2)*r['height']]
            if not all(math.isfinite(v) for v in box) or max(abs(a-b) for a,b in zip(box,obj['bbox_xyxy'])) > 0.001:
                raise ValueError(f'Label differs from review: {label}')
        expected_images.add(image)
        expected_labels.add(label.resolve())
    actual_images = {p.resolve() for p in (root/'images').rglob('*') if p.is_file() and p.suffix.lower() in {'.png','.jpg','.jpeg','.webp','.bmp'}}
    actual_labels = {p.resolve() for p in (root/'labels').rglob('*.txt')}
    if actual_images != expected_images or actual_labels != expected_labels:
        raise ValueError('Extra or missing dataset image/label files')
