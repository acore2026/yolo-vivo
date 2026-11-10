"""Compare the serving GPU path with the unchanged CPU reference, no live frames."""
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'train'))
import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
from service.flower_runtime import FlowerRuntime
from service.flower_inference import features_on_device, classify_boxes
from flower_classifier import features, crop_image
from three_flower_cascade import keep


def sync():
    torch.cuda.synchronize()


def main():
    runtime = FlowerRuntime()
    runtime.prepare()
    assert runtime.classifier_device.startswith('cuda:'), runtime.status()
    det, enc, pre, head = runtime.models
    source = Image.open(ROOT / 'test_videos/test.JPG').convert('RGB')
    images = [ImageOps.fit(source, (480, 640)), source.resize((640, 480)),
              Image.new('RGB', (480, 640))]
    for name in ('test.MOV', 'test2.MOV'):
        cap = cv2.VideoCapture(str(ROOT / 'test_videos' / name))
        try:
            for frame_id in (0, 60, 120):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ok, frame = cap.read()
                if ok:
                    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    image.thumbnail((960, 720))
                    images.append(image)
        finally:
            cap.release()
    cases = []
    # Detector outputs are shared for an exact CPU/GPU classifier comparison.
    for image in images:
        boxes = det.predict(np.asarray(image)[:, :, ::-1].copy(), device=runtime.device,
                            imgsz=640, conf=.01, iou=.7, verbose=False)[0].boxes.data.cpu().tolist()
        cases.append((image, boxes))
    enc.cpu(); head.cpu()
    references = []
    for image, boxes in cases:
        f = features(enc, pre, [crop_image(image, box[:4]) for box in boxes])
        with torch.inference_mode():
            scores = head(f).softmax(1)
        rows = []
        for box, score in zip(boxes, scores.tolist()):
            cls = int(np.argmax(score))
            rows.append({'class': cls, 'xyxy': box[:4], 'confidence': box[4] * score[cls],
                         'detector_confidence': box[4], 'classifier_confidence': score[cls]})
        references.append((f, scores, rows))
    crops = [crop_image(cases[0][0], b[:4]) for b in cases[0][1]]
    assert crops, 'Need detected flowers for performance test'
    cpu_ms = []
    for _ in range(6):
        start = time.perf_counter(); features(enc, pre, crops)
        cpu_ms.append((time.perf_counter() - start) * 1000)
    # Verify multiple batches (>16) and the empty case as well as normal scenes.
    many = (crops * 20)[:20]
    ref_many = features(enc, pre, many)
    enc.to('cuda:0'); head.to('cuda:0')
    max_feature_error = max_score_error = 0.
    for (image, boxes), (ref_f, ref_scores, ref_rows) in zip(cases, references):
        f = features_on_device(enc, pre, [crop_image(image, b[:4]) for b in boxes])
        with torch.inference_mode():
            scores = head(f).softmax(1).cpu()
        torch.testing.assert_close(f.cpu(), ref_f, atol=2e-5, rtol=2e-4)
        torch.testing.assert_close(scores, ref_scores, atol=2e-4, rtol=2e-4)
        assert scores.argmax(1).tolist() == ref_scores.argmax(1).tolist()
        rows = classify_boxes(enc, pre, head, image, boxes)
        old = keep(ref_rows, .05, .7); new = keep(rows, .05, .7)
        assert [(r['class'], r['xyxy']) for r in old] == [(r['class'], r['xyxy']) for r in new]
        if f.numel():
            max_feature_error = max(max_feature_error, (f.cpu()-ref_f).abs().max().item())
            max_score_error = max(max_score_error, (scores-ref_scores).abs().max().item())
    torch.testing.assert_close(features_on_device(enc, pre, many).cpu(), ref_many, atol=2e-5, rtol=2e-4)
    empty = features_on_device(enc, pre, [])
    assert empty.shape == (0, 512) and empty.device.type == 'cuda'
    assert classify_boxes(enc, pre, head, images[0], []) == []
    gpu_ms = []
    for _ in range(8):
        sync(); start = time.perf_counter(); features_on_device(enc, pre, crops); sync()
        gpu_ms.append((time.perf_counter() - start) * 1000)
    assert statistics.median(gpu_ms[1:]) < statistics.median(cpu_ms[1:])
    report = dict(passed=True, cases=len(cases), candidate_counts=[len(b) for _, b in cases],
                  model=runtime.status(), retained_classes_and_boxes_match=True,
                  max_feature_abs_error=max_feature_error, max_score_abs_error=max_score_error,
                  clip_cpu_median_ms=statistics.median(cpu_ms[1:]),
                  clip_gpu_median_ms=statistics.median(gpu_ms[1:]),
                  clip_gpu_samples_ms=gpu_ms, clip_cpu_samples_ms=cpu_ms,
                  precision='float32', empty_and_20_crop_batches_checked=True)
    output = ROOT / 'runtime/clip-gpu-verification.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
