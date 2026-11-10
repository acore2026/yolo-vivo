"""Frozen-weight, validation-only ordinary YOLO resolution comparison."""
import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from common import ROOT, resolve, sha
from audit_validation import match, panel
from standard_yolo import preflight, write_json

SIZES = [640, 960, 1280]
THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8]


def score(records, conf):
    stats = [dict(class_id=c, tp=0, fp=0, fn=0) for c in range(4)]
    sizes = {key: {'gt': 0, 'tp': 0} for key in ['under_96', '96_to_160', 'over_160']}
    errors = []
    for row in records:
        preds = [b for b in row['predictions'] if b['confidence'] >= conf]
        m = match(row['gt'], preds)
        for e in m['predictions']:
            stats[preds[e['prediction']]['class']]['tp' if e['kind'] == 'tp' else 'fp'] += 1
        for i in m['missed_gt']:
            stats[row['gt'][i]['class']]['fn'] += 1
        for i, g in enumerate(row['gt']):
            x1, y1, x2, y2 = g['xyxy']
            side = ((x2-x1)*(y2-y1))**0.5 * 640 / max(row['width'], row['height'])
            group = 'under_96' if side < 96 else '96_to_160' if side < 160 else 'over_160'
            sizes[group]['gt'] += 1
            sizes[group]['tp'] += int(i not in m['missed_gt'])
        if m['missed_gt'] or any(e['kind'] != 'tp' for e in m['predictions']):
            errors.append({'id': row['id'], 'predictions': preds, 'matching': m})
    for s in stats:
        s.update(precision=s['tp']/max(1, s['tp']+s['fp']), recall=s['tp']/max(1, s['tp']+s['fn']))
    for s in sizes.values():
        s['recall'] = s['tp']/s['gt'] if s['gt'] else None
    minimum = min(s[k] for s in stats for k in ['precision', 'recall'])
    return {'conf': conf, 'stats': stats, 'minimum_pr': minimum, 'passes': minimum > 0.9,
            'macro_f1': sum(2*s['tp']/max(1, 2*s['tp']+s['fp']+s['fn']) for s in stats)/4,
            'size_groups': sizes, 'size_definition': 'Custom sqrt(box area) at reference 640 long side; not COCO size bins',
            'errors': errors}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workflow', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    workflow, out = resolve(a.workflow), resolve(a.output)
    state = json.loads((workflow/'status.json').read_text())
    assert state['status'] == 'completed' and not state['cpu_smoke_only']
    rows, checks = preflight(state['configuration'])
    rows = [r for r in rows if r['split'] == 'val']
    assert len(rows) == 39
    dataset = resolve(state['configuration']['dataset'])
    for r in rows:
        for key in ['image', 'label']:
            assert sha(dataset/r[key]) == r[key+'_sha256']
    checkpoint = resolve(state['best_checkpoint'])
    checkpoint_hash = sha(checkpoint)
    assert checkpoint_hash == state['best_checkpoint_sha256']
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'experiment_plan.json', {'checkpoint': str(checkpoint), 'checkpoint_sha256': checkpoint_hash,
               'sizes': SIZES, 'conf_grid': THRESHOLDS, 'fixed_conf': 0.2, 'nms_iou': 0.5,
               'agnostic_nms': True, 'max_det': 100, 'split': 'val', 'selection': 'max minimum per-class P/R, then macro-F1, then smaller input',
               'retraining': False, 'test_used': False, 'preflight': checks,
               'manifest_sha256': sha(dataset/'manifest.json')})
    os.environ['YOLO_CONFIG_DIR'] = str(ROOT/'train/.cache/standard_yolo')
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    torch.set_num_threads(4)
    from ultralytics import YOLO
    model = YOLO(str(checkpoint))
    assert model.model.__class__.__name__ == 'DetectionModel'
    results, all_records = {}, {}
    for size in SIZES:
        # Exclude one warm-up inference from wall-time measurements at each size.
        model.predict(str(dataset/rows[0]['image']), imgsz=size, conf=0.001, iou=0.5, agnostic_nms=True, max_det=100, device='cpu', verbose=False)
        records, durations = [], []
        for r in rows:
            start = time.perf_counter()
            b = model.predict(str(dataset/r['image']), imgsz=size, conf=0.001, iou=0.5, agnostic_nms=True, max_det=100, device='cpu', verbose=False)[0].boxes.data.cpu()
            durations.append((time.perf_counter()-start)*1000)
            assert torch.isfinite(b).all()
            records.append({'id': r['id'], 'image': r['image'], 'image_sha256': r['image_sha256'],
                            'width': r['width'], 'height': r['height'],
                            'gt': [{'class': x['class_id'], 'xyxy': x['bbox_xyxy']} for x in r['objects']],
                            'predictions': [{'class': int(x[5]), 'confidence': x[4], 'xyxy': x[:4]} for x in b.tolist()]})
        trials = [score(records, c) for c in THRESHOLDS]
        best = max(trials, key=lambda t: (t['minimum_pr'], t['macro_f1']))
        entry = {'imgsz': size, 'fixed_conf': score(records, 0.2), 'selected': best,
                 'cpu_wall_ms_mean': sum(durations)/len(durations), 'cpu_wall_ms_median': float(np.median(durations)),
                 'timing_scope': '4 CPU threads, single-image predict incl file load, no first warmup; not GPU benchmark'}
        results[str(size)] = entry
        all_records[size] = records
        write_json(out/f'predictions_{size}.json', records)
        write_json(out/f'thresholds_{size}.json', trials)
        write_json(out/'comparison.json', results)
        print(json.dumps({'imgsz': size, 'fixed_conf_stats': entry['fixed_conf']['stats'],
                          'selected_conf': best['conf'], 'selected_stats': best['stats'],
                          'passes': best['passes'], 'cpu_wall_ms_mean': entry['cpu_wall_ms_mean']}), flush=True)
    selected = max(results.values(), key=lambda x: (x['selected']['minimum_pr'], x['selected']['macro_f1'], -x['imgsz']))
    point = {k: v for k, v in selected['selected'].items() if k != 'errors'}
    point.update(imgsz=selected['imgsz'], checkpoint_sha256=checkpoint_hash, nms_iou=0.5, agnostic_nms=True,
                 match_iou=0.5, scope='Selected using 39 development val images only; not independent acceptance',
                 selection_manifest_sha256=sha(dataset/'manifest.json'))
    write_json(out/'operating_point.json', point)
    # Concrete validation errors at the selected operating point, plus original 640 errors.
    ids = list(dict.fromkeys(e['id'] for e in results['640']['fixed_conf']['errors'] + selected['selected']['errors']))[:12]
    preview = out/'review'; preview.mkdir()
    for image_id in ids:
        base = next(r for r in all_records[640] if r['id'] == image_id)
        chosen = next(r for r in all_records[selected['imgsz']] if r['id'] == image_id)
        im = cv2.imread(str(dataset/base['image']))
        panels = [panel(im, base['gt'], image_id+' | existing reviewed GT'),
                  panel(im, [b for b in base['predictions'] if b['confidence']>=0.2], '640 conf .20'),
                  panel(im, [b for b in chosen['predictions'] if b['confidence']>=point['conf']], f"{point['imgsz']} conf {point['conf']}")]
        assert cv2.imwrite(str(preview/(image_id+'.jpg')), np.hstack(panels))
    write_json(preview/'index.json', {'image_ids': ids, 'new_annotation_review': False, 'selection':'validation errors; not all validation images'})
    assert sha(checkpoint) == checkpoint_hash
    print('Completed', out, 'selected', point['imgsz'], point['conf'], 'passes', point['passes'], flush=True)


if __name__ == '__main__':
    main()
