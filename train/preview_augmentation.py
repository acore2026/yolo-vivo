"""Preview the actual training pipeline and check labels before/after closing Mosaic."""
import argparse
from copy import deepcopy
import json
import random
from pathlib import Path
import cv2
import numpy as np
import torch
from common import config, resolve, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='train/configs/flowers_augmented.yaml')
    parser.add_argument('--output', default='train/diagnostics/augmentation_preview_v1')
    args = parser.parse_args()
    from runtime import setup
    setup()
    from ultralytics.cfg import get_cfg
    from flower_augmentation import build_flower_dataset, CameraDegradation
    c = config(args.config)
    out = resolve(args.output)
    out.mkdir(parents=True, exist_ok=False)
    d = config(c['data'])
    d['names'] = dict(enumerate(d['names']))
    d.update(nc=len(d['names']), channels=3)
    hyp = get_cfg(overrides=c['training'])
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    dataset = build_flower_dataset(hyp, str(Path(d['path'])/d['train']), 8, d)
    manifest = resolve(c['data']).parent/'manifest.json'
    initial_hash = sha(manifest)
    records = []
    tiles = []
    for phase in ('normal', 'mosaic_closed'):
        if phase == 'mosaic_closed':
            closed_hyp = deepcopy(hyp)
            dataset.close_mosaic(closed_hyp)
            assert closed_hyp.mosaic == 0.0
        assert sum(isinstance(t, CameraDegradation) for t in dataset.transforms.transforms) == 1
        for i in range(len(dataset)):
            item = dataset[i]
            boxes = item['bboxes'].numpy()
            classes = item['cls'].numpy().reshape(-1).astype(int)
            assert np.isfinite(boxes).all() and ((boxes >= 0) & (boxes <= 1)).all()
            assert (boxes[:, 2:] > 0).all() and len(boxes) == len(classes)
            for cls in classes:
                assert item['texts'][cls] in d['names'].values()
            records.append({'phase':phase,'index':i,'primary_source':item['im_file'],
                            'boxes':len(boxes), 'texts':item['texts']})
            if phase == 'normal' and i % 20 == 0:
                img = item['img'].permute(1,2,0).numpy()[:, :, ::-1].copy()
                h,w = img.shape[:2]
                for box,cls in zip(boxes, classes):
                    x,y,bw,bh = box
                    p1=(int((x-bw/2)*w),int((y-bh/2)*h))
                    p2=(int((x+bw/2)*w),int((y+bh/2)*h))
                    cv2.rectangle(img,p1,p2,(0,220,255),2)
                    cv2.putText(img,item['texts'][cls],(max(0,p1[0]),max(15,p1[1])),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,255),1)
                cv2.putText(img,f'sample {i}: {Path(item["im_file"]).stem}',(8,20),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
                cv2.imwrite(str(out/f'sample_{i:03d}.jpg'),img)
                tiles.append(cv2.resize(img,(400,400)))
    while len(tiles)%4:
        tiles.append(np.zeros((400,400,3),dtype=np.uint8))
    cv2.imwrite(str(out/'contact_sheet.jpg'),np.vstack([np.hstack(tiles[i:i+4]) for i in range(0,len(tiles),4)]))
    # The same custom dataset must leave validation free of random camera transforms.
    dataset.augment = False
    validation_transforms = dataset.build_transforms(deepcopy(hyp))
    assert not any(isinstance(t,CameraDegradation) for t in validation_transforms.transforms)
    assert sha(manifest) == initial_hash
    report={'result':'PASS','seed':42,'checked_samples':len(records),
            'preview_images':14,'manifest_sha256':initial_hash,'config_sha256':sha(resolve(args.config)),
            'checks':['normalized finite boxes','text/class correspondence',
                      'camera transform survives close_mosaic','no camera transform in validation',
                      'manifest unchanged'],
            'note':'Preview uses the real randomized pipeline; primary_source is not all Mosaic sources. No new training labels exported.',
            'samples':records}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(f'PASS: {len(records)} samples; preview: {out}')


if __name__ == '__main__':
    main()
