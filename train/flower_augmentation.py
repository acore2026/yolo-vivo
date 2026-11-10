"""Train-only camera degradation; geometry/box transforms remain Ultralytics-owned."""
import random
import cv2
import numpy as np
from ultralytics.data.dataset import YOLOMultiModalDataset


class CameraDegradation:
    def __call__(self, labels):
        img = labels['img']
        if random.random() < 0.10:
            img = cv2.GaussianBlur(img, (3, 3), random.uniform(0.3, 0.8))
        if random.random() < 0.10:
            noise = np.random.normal(0, random.uniform(1, 4), img.shape)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        if random.random() < 0.10:
            ok, encoded = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, random.randint(75, 95)])
            if not ok:
                raise RuntimeError('JPEG augmentation encoding failed')
            img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        labels['img'] = img
        return labels


class FlowerAugmentedDataset(YOLOMultiModalDataset):
    def build_transforms(self, hyp):
        transforms = super().build_transforms(hyp)
        if self.augment:
            transforms.insert(-1, CameraDegradation())
        return transforms


def build_flower_dataset(cfg, img_path, batch, data, stride=32):
    if cfg.fraction != 1.0:
        raise ValueError('Augmentation comparison requires the complete training split')
    return FlowerAugmentedDataset(
        img_path=img_path, imgsz=cfg.imgsz, batch_size=batch, augment=True,
        hyp=cfg, rect=cfg.rect, cache=cfg.cache or None, single_cls=cfg.single_cls,
        stride=stride, pad=0.0, prefix='flower augmentation: ', task=cfg.task,
        classes=cfg.classes, data=data, fraction=1.0,
    )
