"""Ordinary four-class YOLO: checked data -> smoke -> fresh pretrained training -> val.

No CLIP, LoRA, test-based tuning, deployment, or original-weight replacement.
CPU smoke uses random initialization and is NOT an accuracy experiment.
"""
import argparse
from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
import urllib.request

import yaml
from common import ROOT, resolve, sha, validate_training_labels

CONFIG = ROOT / 'train/configs/flowers_standard_yolo.yaml'
WEIGHT_URL = 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt'


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def preflight(c):
    source = resolve(c['source_dataset'])
    rows = json.loads((source / 'manifest.json').read_text())
    counts = dict(Counter(r['split'] for r in rows))
    assert counts == c.get('expected_images', {'train': 280, 'val': 39}), counts
    if c.get('source_manifest_sha256'):
        assert sha(source / 'manifest.json') == c['source_manifest_sha256']
    groups, videos, frames = {}, {}, set()
    by_id = {r['id']: r for r in rows}
    assert len(by_id) == len(rows), 'Duplicate sample IDs'
    for r in rows:
        assert r['status'] == 'accepted' and r['review']['reviewer'] == 'assistant_visual'
        crop_key = None
        if not r['objects']:
            # Empty labels are allowed only for explicit, visually reviewed,
            # reproducible crops from a positive image in this training split.
            assert c.get('allow_reviewed_background_crops') is True
            assert r['split'] == 'train' and r.get('negative_verified') is True
            assert r['review']['decision'] == 'accepted_negative' and r['review']['no_target_visible'] is True
            assert r['annotation_type'] == 'reviewed_background_crop'
            parent = by_id[r['parent_id']]
            assert parent['split'] == 'train' and parent['objects']
            assert parent['source_sha256'] == r['source_sha256'] and parent['frame'] == r['frame']
            assert parent['image_sha256'] == r['parent_image_sha256']
            assert parent['image'] == r['parent_image'] and parent['group'] == r['group']
            crop_key = tuple(r['crop_xyxy'])
            x1, y1, x2, y2 = crop_key
            assert all(isinstance(v, int) for v in crop_key)
            assert 0 <= x1 < x2 <= parent['width'] and 0 <= y1 < y2 <= parent['height']
            assert (x2-x1, y2-y1) == (r['width'], r['height'])
            from PIL import Image
            with Image.open(source / parent['image']) as original, Image.open(source / r['image']) as derived:
                assert derived.size == (r['width'], r['height'])
                assert original.crop(crop_key).convert('RGB').tobytes() == derived.convert('RGB').tobytes()
            assert all(x2 < b['bbox_xyxy'][0]-20 or x1 > b['bbox_xyxy'][2]+20 or y2 < b['bbox_xyxy'][1]-20 or y1 > b['bbox_xyxy'][3]+20 for b in parent['objects'])
        for kind in ['image', 'label']:
            assert sha(source / r[kind]) == r[kind + '_sha256'], r['id']
        sample_key = (r['source_sha256'], r['frame'], crop_key)
        assert sample_key not in frames
        frames.add(sample_key)
        videos.setdefault(r['source_sha256'], set()).add(r['split'])
        groups.setdefault(r['group'], set()).add(r['split'])
        for b in r['objects']:
            x1, y1, x2, y2 = b['bbox_xyxy']
            # Existing serialized YOLO coordinates have sub-pixel roundoff.
            assert b['class_id'] in range(4)
            assert -0.001 <= x1 < x2 <= r['width'] + 0.001
            assert -0.001 <= y1 < y2 <= r['height'] + 0.001
    assert all(len(x) == 1 for x in videos.values()), 'Source-video split leakage'
    image_splits = {}
    for r in rows:
        image_splits.setdefault(r['image_sha256'], set()).add(r['split'])
    assert all(len(x) == 1 for x in image_splits.values()), 'Duplicate-image leakage'
    validate_training_labels(rows, source)
    return rows, {'images': counts,
                  'reviewed_background_crops': sum(not r['objects'] for r in rows),
                  'objects': {s: dict(Counter(o['class_id'] for r in rows if r['split'] == s for o in r['objects'])) for s in ['train', 'val']},
                  'source_manifest_sha256': sha(source / 'manifest.json'),
                  'source_video_isolated': True, 'independent_scene_split': False,
                  'review': 'existing assistant_visual, not human acceptance',
                  'all_source_image_label_hashes': 'PASS', 'test_included': False}


def prepare_data(c, rows, smoke=False, folder=None):
    source = resolve(c['source_dataset'])
    target = folder / 'data' if smoke else resolve(c['dataset'])
    chosen = []
    for split, count in [('train', 16), ('val', 8)]:
        subset = [r for r in rows if r['split'] == split]
        chosen += [subset[round(i * (len(subset) - 1) / (count - 1))] for i in range(count)] if smoke else subset
    # Separate copies isolate framework label caches and original annotations.
    if not target.exists():
        target.mkdir(parents=True, exist_ok=False)
        for r in chosen:
            for key in ['image', 'label']:
                dest = target / r[key]
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / r[key], dest)
        write_json(target / 'manifest.json', chosen)
        (target / 'data.yaml').write_text(yaml.safe_dump({'path': str(target), 'train': 'images/train', 'val': 'images/val', 'names': c['names']}, sort_keys=False))
    assert json.loads((target / 'manifest.json').read_text()) == chosen
    for r in chosen:
        for key in ['image', 'label']:
            assert sha(target / r[key]) == r[key + '_sha256']
    validate_training_labels(chosen, target)
    data = yaml.safe_load((target / 'data.yaml').read_text())
    assert data['names'] == c['names'] and 'test' not in data
    return target / 'data.yaml', chosen


def fetch_weights(path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.download')
    if temporary.exists():
        raise RuntimeError(f'Incomplete download already exists: {temporary}')
    try:
        with urllib.request.urlopen(WEIGHT_URL, timeout=45) as response, temporary.open('xb') as dest:
            shutil.copyfileobj(response, dest)
        if temporary.stat().st_size < 1000000:
            raise RuntimeError('Downloaded checkpoint is unexpectedly small')
        temporary.rename(path)
        write_json(path.with_suffix('.source.json'), {'url': WEIGHT_URL, 'sha256': sha(path), 'note': 'Local provenance hash, not an independent publisher checksum'})
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def evaluate_val(model, rows, data, output, device):
    from audit_validation import match
    records = []
    for r in rows:
        if r['split'] != 'val':
            continue
        result = model.predict(str(data.parent / r['image']), device=device, imgsz=640, conf=0.001, iou=0.5, agnostic_nms=True, max_det=100, verbose=False)[0]
        pred = [{'class': int(b[5]), 'xyxy': b[:4], 'confidence': b[4]} for b in result.boxes.data.cpu().tolist()]
        records.append({'id': r['id'], 'gt': [{'class': b['class_id'], 'xyxy': b['bbox_xyxy']} for b in r['objects']], 'predictions': pred})
    trials = []
    for threshold in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8]:
        stats = [dict(class_id=i, tp=0, fp=0, fn=0) for i in range(4)]
        for r in records:
            pred = [p for p in r['predictions'] if p['confidence'] >= threshold]
            result = match(r['gt'], pred)
            for event in result['predictions']:
                stats[pred[event['prediction']]['class']]['tp' if event['kind'] == 'tp' else 'fp'] += 1
            for index in result['missed_gt']:
                stats[r['gt'][index]['class']]['fn'] += 1
        for s in stats:
            s.update(precision=s['tp'] / max(1, s['tp'] + s['fp']), recall=s['tp'] / max(1, s['tp'] + s['fn']))
        trials.append({'conf': threshold, 'stats': stats, 'minimum_pr': min(s[k] for s in stats for k in ['precision', 'recall']), 'macro_f1': sum(2*s['tp']/max(1,2*s['tp']+s['fp']+s['fn']) for s in stats)/4})
    best = max(trials, key=lambda t: (t['minimum_pr'], t['macro_f1']))
    best.update(nms_iou=0.5, agnostic_nms=True, match_iou=0.5, imgsz=640, passes=best['minimum_pr'] > 0.9, scope='Selected on development val; not independent test')
    write_json(output / 'val_predictions.json', records)
    write_json(output / 'val_thresholds.json', trials)
    write_json(output / 'operating_point.json', best)
    return best


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--cpu-smoke', action='store_true')
    mode.add_argument('--check-only', action='store_true')
    p.add_argument('--config', default=str(CONFIG), help='Reviewed dataset and training configuration')
    p.add_argument('--fetch-weights', action='store_true', help='Download missing official pretrained weights before GPU training')
    a = p.parse_args()
    c = yaml.safe_load(resolve(a.config).read_text())
    rows, report = preflight(c)
    os.environ['YOLO_CONFIG_DIR'] = str(ROOT / 'train/.cache/standard_yolo')
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    import torch
    import importlib.metadata
    torch.set_num_threads(4)
    assert importlib.metadata.version('ultralytics') == '8.4.183', 'Recheck workflow with this Ultralytics version'
    report.update(cuda_available=torch.cuda.is_available(), pretrained_weights_present=resolve(c['model']).is_file())
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not a.run and not a.cpu_smoke:
        return
    if a.run:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable here. Run this same command in the GPU-enabled WSL terminal.')
        x = torch.ones((32, 32), device='cuda:0')
        assert (x @ x).mean().item() == 32
        if a.fetch_weights:
            fetch_weights(resolve(c['model']))
        if not resolve(c['model']).is_file():
            raise FileNotFoundError('Missing models/yolov8s.pt; use --fetch-weights in a network-enabled terminal')
    from ultralytics import YOLO, settings
    integrations = ['sync', 'tensorboard', 'wandb', 'mlflow', 'comet', 'dvc', 'clearml', 'neptune']
    settings.update({key: False for key in integrations if key in settings})
    folder = ROOT / 'train/workflows' / (c.get('workflow_prefix', 'standard_yolo') + '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    folder.mkdir(parents=True, exist_ok=False)
    state = {'status': 'running', 'cpu_smoke_only': a.cpu_smoke, 'preflight': report, 'configuration': c, 'test_used': False}
    write_json(folder / 'status.json', state)
    print('Workflow:', folder, flush=True)
    try:
        data, chosen = prepare_data(c, rows, smoke=a.cpu_smoke, folder=folder)
        base = 'yolov8s.yaml' if a.cpu_smoke else str(resolve(c['model']))
        if a.run:
            state['base_sha256'] = sha(Path(base))
        stages = [('cpu_architecture_smoke', 1)] if a.cpu_smoke else [('smoke', 1), ('full', c['training']['epochs'])]
        for stage, epochs in stages:
            state['stage'] = stage
            write_json(folder / 'status.json', state)
            model = YOLO(base, task='detect')
            assert model.model.__class__.__name__ == 'DetectionModel', 'Expected ordinary YOLO, not YOLO-World'
            kwargs = dict(c['training'])
            kwargs.update(epochs=epochs, data=str(data), project=str(folder), name=stage, exist_ok=False, plots=True, save=True, val=True)
            if stage != 'full':
                kwargs.update(patience=0, close_mosaic=0)
            if a.cpu_smoke:
                kwargs.update(device='cpu', workers=0, batch=2, imgsz=320, pretrained=False)
            model.train(**kwargs)
            checkpoint = folder / stage / 'weights/best.pt'
            assert checkpoint.is_file()
            reloaded = YOLO(str(checkpoint))
            assert list(reloaded.names.values()) == c['names']
            result = reloaded.predict(str(data.parent / chosen[0]['image']), device='cpu' if a.cpu_smoke else c['training']['device'], imgsz=320 if a.cpu_smoke else 640, verbose=False)[0]
            assert torch.isfinite(result.boxes.data).all()
            if stage == 'full':
                state['validation'] = evaluate_val(reloaded, chosen, data, folder, c['training']['device'])
                state['best_checkpoint'] = str(checkpoint)
                state['best_checkpoint_sha256'] = sha(checkpoint)
                point = json.loads((folder / 'operating_point.json').read_text())
                point['checkpoint_sha256'] = state['best_checkpoint_sha256']
                write_json(folder / 'operating_point.json', point)
            state.setdefault('completed', []).append(stage)
            write_json(folder / 'status.json', state)
        preflight(c)
        if a.run:
            assert sha(Path(base)) == state['base_sha256']
        state['status'] = 'cpu_smoke_passed_not_trained_baseline' if a.cpu_smoke else 'completed'
        write_json(folder / 'status.json', state)
        print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)
    except BaseException as exc:
        state.update(status='failed', error=str(exc))
        write_json(folder / 'status.json', state)
        (folder / 'error.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
