"""Validation-only threshold calibration, with a fixed nested-box alternative.

No boundary exclusions, image-specific rules, relabeling, retraining, or test reads.
"""
import argparse
import copy
import json

from common import resolve, sha
from standard_yolo import write_json
from compare_standard_resolutions import score, THRESHOLDS
from standard_flower_postprocess import filter_predictions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resolution-run', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    source, out = resolve(a.resolution_run), resolve(a.output)
    selected = json.loads((source/'operating_point.json').read_text())
    candidates = source/f"predictions_{selected['imgsz']}.json"
    rows = json.loads(candidates.read_text())
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'experiment_plan.json', {'prediction_source': str(candidates), 'prediction_sha256': sha(candidates),
               'imgsz': selected['imgsz'], 'split': 'val', 'conf_grid': THRESHOLDS,
               'alternatives': ['per_class_threshold_only', 'per_class_threshold_with_same_class_containment_0.9'],
               'selection': 'per-class max min(P,R), then F1; joint verification; select max overall minimum P/R then macro-F1, prefer no suppression on ties',
               'test_used': False, 'changes_to_annotations': False,
               'risk': 'nested suppression can discard a distinct same-class pot if its box is mostly contained within another'})
    results = []
    for containment in [None, 0.9]:
        scanned = []
        for threshold in THRESHOLDS:
            altered = copy.deepcopy(rows)
            for r in altered:
                r['predictions'] = filter_predictions(r['predictions'], threshold, containment)
            scanned.append(score(altered, 0.0) | {'threshold': threshold})
        thresholds = []
        for c in range(4):
            def rank(t):
                s = t['stats'][c]
                return min(s['precision'], s['recall']), 2*s['tp']/max(1, 2*s['tp']+s['fp']+s['fn'])
            thresholds.append(max(scanned, key=rank)['threshold'])
        joint = copy.deepcopy(rows)
        for r in joint:
            r['predictions'] = filter_predictions(r['predictions'], thresholds, containment)
        result = score(joint, 0.0)
        result.update(conf=thresholds, same_class_containment=containment)
        results.append(result)
        tag = 'threshold_only' if containment is None else 'nested_0p9'
        write_json(out/f'{tag}_scan.json', scanned)
        write_json(out/f'{tag}_joint.json', joint)
    best = max(results, key=lambda r: (r['minimum_pr'], r['macro_f1'], r['same_class_containment'] is None))
    point = {k: v for k, v in best.items() if k != 'errors'}
    point.update(imgsz=selected['imgsz'], checkpoint_sha256=selected['checkpoint_sha256'],
                 nms_iou=selected['nms_iou'], agnostic_nms=True, match_iou=0.5,
                 scope='Postprocessing selected on existing development val; not independent acceptance',
                 prediction_source_sha256=sha(candidates))
    write_json(out/'comparison.json', results)
    write_json(out/'operating_point.json', point)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
