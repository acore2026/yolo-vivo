"""Fixed-parameter ordinary YOLO inference; preserves all previous outputs."""
import argparse
import json
import os
from pathlib import Path

import cv2
import torch

from common import ROOT, resolve, sha
from standard_flower_postprocess import filter_predictions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workflow', required=True)
    p.add_argument('--sources', nargs='+', default=['test_videos/test.JPG', 'test_videos/test.MOV'])
    p.add_argument('--output', required=True)
    p.add_argument('--operating-point', help='Optional independently saved validation calibration, with matching checkpoint hash')
    a = p.parse_args()
    folder = resolve(a.workflow)
    state = json.loads((folder / 'status.json').read_text())
    assert state['status'] == 'completed' and not state['cpu_smoke_only']
    checkpoint = resolve(state['best_checkpoint'])
    digest = sha(checkpoint)
    point_path = resolve(a.operating_point) if a.operating_point else folder / 'operating_point.json'
    point = json.loads(point_path.read_text())
    assert digest == state['best_checkpoint_sha256'] == point['checkpoint_sha256']
    sources = [resolve(s) for s in a.sources]
    for source in sources:
        assert source.is_file(), source
    out = resolve(a.output)
    out.mkdir(parents=True, exist_ok=False)
    os.environ['YOLO_CONFIG_DIR'] = str(ROOT / 'train/.cache/standard_yolo')
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    torch.set_num_threads(4)
    from ultralytics import YOLO
    model = YOLO(str(checkpoint))
    assert model.model.__class__.__name__ == 'DetectionModel'
    assert list(model.names.values()) == state['configuration']['names']
    summary = {'model_kind': 'ordinary_yolo', 'model_label': 'ordinary YOLOv8s' + (' calibrated' if a.operating_point else ''),
               'model_sha256': digest, 'checkpoint': str(checkpoint),
               'operating_point': point, 'operating_point_source': str(point_path),
               'operating_point_sha256': sha(point_path), 'thresholds_selected_on': 'development validation set',
               'note': 'Fixed before user-test inference; predictions are not accuracy; videos contain no audio',
               'inputs': []}
    (out / 'frozen_inference_config.json').write_text(json.dumps(summary, indent=2))
    for index, source in enumerate(sources, 1):
        isvideo = source.suffix.lower() in {'.mov', '.mp4', '.avi', '.mkv'}
        cap = cv2.VideoCapture(str(source)) if isvideo else None
        if isvideo:
            assert cap.isOpened()
        expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap is not None else 1
        fps = cap.get(cv2.CAP_PROP_FPS) if cap is not None else None
        target = out / f'{index:02d}_{source.stem}{".mp4" if isvideo else ".jpg"}'
        writer = None
        n = 0
        with (out / f'{index:02d}_{source.stem}.jsonl').open('x') as log:
            try:
                while True:
                    if cap is not None:
                        ok, frame = cap.read()
                    else:
                        frame = cv2.imread(str(source)) if n == 0 else None
                        ok = frame is not None
                    if not ok:
                        break
                    boxes = model.predict(frame, device='cpu', imgsz=point['imgsz'],
                                          conf=0.001 if isinstance(point['conf'], list) else point['conf'], iou=point['nms_iou'],
                                          agnostic_nms=point['agnostic_nms'], max_det=100,
                                          verbose=False)[0].boxes.data.cpu()
                    assert torch.isfinite(boxes).all()
                    pred = [{'class': int(b[5]), 'xyxy': b[:4], 'confidence': b[4]} for b in boxes.tolist()]
                    pred = filter_predictions(pred, point['conf'], point.get('same_class_containment'))
                    log.write(json.dumps({'frame': n, 'detections': pred}) + '\n')
                    size = max(0.6, frame.shape[1] / 1600)
                    thickness = max(2, round(size * 2))
                    for b in pred:
                        x1, y1, x2, y2 = [round(v) for v in b['xyxy']]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), thickness)
                        cv2.putText(frame, f"{model.names[b['class']]} {b['confidence']:.2f}",
                                    (max(0, x1), max(round(30*size), y1-8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, size, (0, 80, 240), thickness)
                    if isvideo:
                        if writer is None:
                            assert fps > 0
                            h, w = frame.shape[:2]
                            writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                            assert writer.isOpened()
                        writer.write(frame)
                    else:
                        assert cv2.imwrite(str(target), frame)
                    n += 1
                    if n % 100 == 0:
                        print(source.name, n, expected, flush=True)
            finally:
                if cap is not None:
                    cap.release()
                if writer is not None:
                    writer.release()
        assert n == expected and n > 0
        summary['inputs'].append({'source': str(source.relative_to(ROOT)),
                                  'source_sha256': sha(source), 'frames': n, 'output': str(target)})
    assert sha(checkpoint) == digest
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
