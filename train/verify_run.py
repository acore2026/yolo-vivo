"""Verify trained artifacts and inference on one validation image; never test data."""
import argparse
import csv
import json
import math
from pathlib import Path
from common import config, resolve, sha

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    a=p.parse_args();c=config(a.config);run=resolve(c['project'])/c['name']
    integrity=json.loads((run/'training_integrity.json').read_text())
    for key in ('frozen_parameters_unchanged','batchnorm_unchanged','lora_finite','lora_updated'):
        if integrity.get(key) is not True: raise RuntimeError(f'Integrity check failed: {key}')
    with (run/'results.csv').open() as f: rows=list(csv.DictReader(f))
    if not rows: raise RuntimeError('No epoch metrics')
    for row in rows:
        for key,value in row.items():
            if not math.isfinite(float(value)): raise RuntimeError(f'Nonfinite metric: {key}')
    for name in ('best.pt','last.pt','adapter_best.pt','adapter_last.pt','training_state_last.pt'):
        if not (run/'weights'/name).is_file(): raise RuntimeError(f'Missing checkpoint: {name}')
    import torch
    from model_io import load_model
    from ultralytics import YOLOWorld
    d=config(c['data']);root=Path(d['path']);manifest=json.loads((root/'manifest.json').read_text())
    image=root/next(r['image'] for r in manifest if r['split']=='val')
    reports={}
    for tag,model in [('adapter',load_model(resolve(c['model']),run/'weights/adapter_best.pt')),
                      ('merged',YOLOWorld(str(run/'weights/best.pt')))]:
        model.set_classes(c['prompts'])
        result=model.predict(str(image),device=c['training']['device'],imgsz=c['training']['imgsz'],verbose=False)[0]
        if not torch.isfinite(result.boxes.data).all(): raise RuntimeError(f'{tag}: nonfinite predictions')
        reports[tag]={'load_and_predict':'PASS','detections':len(result.boxes)}
        del model
    report={'result':'PASS','epochs_recorded':len(rows),'integrity':integrity,
            'base_sha256':sha(resolve(c['model'])),'validation_image':str(image),'artifacts':reports,
            'scope':'load/inference integrity only, not an accuracy acceptance threshold'}
    (run/'artifact_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
