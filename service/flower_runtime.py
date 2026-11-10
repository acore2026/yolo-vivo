"""Shared, serialized inference using the retained three-flower cascade."""
import json
import os
from pathlib import Path
import sys
import threading

import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CLASSIFIER = ROOT / 'train/workflows/three_flowers_20261109_160053_774018/classifier'
CLASSIFIER_SHA = 'e4ee07981f5f50af67665310484a8e45987c8aebbc2bb4e02931757c55efc1c6'


class FlowerRuntime:
    def __init__(self):
        self.lock = threading.Lock()
        self.ready = False
        self.device = None
        self.classifier_device = None

    def prepare(self):
        with self.lock:
            if self.ready:
                return
            # Training helpers use sibling imports; reuse their verified loading
            # and crop/filter code so live inference matches offline evaluation.
            train_path = str(ROOT / 'train')
            if train_path not in sys.path:
                sys.path.insert(0, train_path)
            import torch
            from common import sha
            from three_flower_cascade import load, keep
            from .flower_inference import candidates
            torch.set_num_threads(4)
            self.device = os.environ.get('FLOWER_DEVICE', 'auto')
            if self.device == 'auto':
                self.device = '0' if torch.cuda.is_available() else 'cpu'
            classifier_device = os.environ.get('FLOWER_CLASSIFIER_DEVICE', 'auto')
            if classifier_device == 'auto':
                classifier_device = self.device
            if classifier_device.isdecimal():
                classifier_device = 'cuda:' + classifier_device
            classifier_device = torch.device(classifier_device)
            self.point = json.loads((CLASSIFIER / 'operating_point.json').read_text())
            if sha(CLASSIFIER / 'classifier.pt') != CLASSIFIER_SHA or self.point['classifier_sha256'] != CLASSIFIER_SHA:
                raise RuntimeError('Three-flower classifier checksum mismatch')
            if (self.point['detector_conf'], self.point['classifier_conf'], self.point['nms_iou']) != (.05, .7, .5):
                raise RuntimeError('Retained operating point changed; review required')
            self.models = load(CLASSIFIER, self.device)
            _, encoder, _, head = self.models
            # cuDNN otherwise permits TF32 convolutions even with FP32 weights.
            # Match the retained CPU classifier numerically before optimizing precision.
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            # Move both stages and all input/features together. Preserve FP32:
            # migration must not silently introduce half precision or new thresholds.
            encoder.to(device=classifier_device, dtype=torch.float32).eval()
            head.to(device=classifier_device, dtype=torch.float32).eval()
            self.classifier_device = str(next(encoder.parameters()).device)
            self.candidates, self.keep = candidates, keep
            # Warm up outside the per-frame deadline, before accepting streams.
            self._infer(Image.new('RGB', (640, 480)))
            self.ready = True

    def _infer(self, image):
        return self.keep(self.candidates(*self.models, image, self.device),
                         self.point['detector_conf'], self.point['classifier_conf'])

    def infer(self, frame):
        self.prepare()
        with self.lock:
            return self._infer(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    def status(self):
        return dict(ready=self.ready, device=self.device, classifier_device=self.classifier_device,
                    classes=['peony', 'lily', 'chrysanthemum'],
                    classifier_sha256=CLASSIFIER_SHA)


runtime = FlowerRuntime()
