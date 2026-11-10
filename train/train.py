"""Check configuration by default. --run explicitly starts training."""
import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
from common import config, resolve, sha, validate_manifest, validate_training_labels

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='train/configs/flowers.yaml')
    p.add_argument('--run',action='store_true')
    a=p.parse_args();c=config(a.config)
    base=resolve(c['model'])
    if not base.is_file(): raise SystemExit(f'Missing base model: {base}')
    if c.get('three_class_conversion'):
        from three_flower_data import PROMPTS
        if c['prompts'] != PROMPTS: raise SystemExit('Three-class prompt order mismatch')
    elif len(c['prompts'])!=4: raise SystemExit('Expected four prompts')
    print(json.dumps({'mode':'train' if a.run else 'check-only','base':str(base),'sha256':sha(base),'data':c['data']},indent=2))
    if not c['data']:
        if a.run: raise SystemExit('BLOCKED: no approved independent train/val dataset configured')
        print('BLOCKED for training: data not assigned. Static configuration readable.');return
    data_path=resolve(c['data']);d=config(data_path)
    policy=d.get('partition_policy','capture_group')
    if policy!=c.get('partition_policy','capture_group'): raise SystemExit('Config/dataset partition policy mismatch')
    if c.get('three_class_conversion'):
        from three_flower_data import verify
        rows, report = verify(data_path.parent)
        counts = report['images']
    else:
        rows,counts,groups=validate_manifest(data_path.parent/'manifest.json',policy)
    validate_training_labels(rows, data_path.parent)
    if not {'train','val'}<=set(counts) or 'review' in counts:
        raise SystemExit('Independent train/val capture groups required')
    if d['names']!=c['prompts']: raise SystemExit('Dataset prompt order mismatch')
    for split in counts:
        expected=(data_path.parent/'images'/split).resolve()
        actual=(Path(d['path'])/d[split]).resolve()
        if actual!=expected: raise SystemExit('Dataset must use prepared manifest image directories')
    if not a.run: print('Dataset preflight passed');return
    if importlib.metadata.version('ultralytics')!='8.4.183': raise SystemExit('Trainer reviewed for ultralytics 8.4.183; revalidate hooks for other versions')
    if importlib.util.find_spec('clip') is None: raise SystemExit('CLIP is missing; installation requires separate approval')
    run=resolve(c['project'])/c['name']
    if run.exists(): raise SystemExit('Choose a new run name; overwriting is disabled')
    os.environ['YOLO_CONFIG_DIR']=str(resolve('train/.cache/ultralytics'))
    os.environ['YOLO_AUTOINSTALL']='false'
    import torch
    if c['training']['device']!='cpu' and not torch.cuda.is_available(): raise SystemExit('CUDA unavailable')
    from runtime import setup
    setup()
    from ultralytics import YOLOWorld
    from lora import LoRAConfig
    from trainer import FlowerLoRATrainer
    FlowerLoRATrainer.adapter_config=LoRAConfig(**c['lora'])
    FlowerLoRATrainer.base_sha256=sha(base)
    FlowerLoRATrainer.camera_augmentation=c.get('camera_augmentation', False)
    model=YOLOWorld(str(base))
    model.train(trainer=FlowerLoRATrainer,data=str(data_path),project=str(resolve(c['project'])),name=c['name'],**c['training'])
    if sha(base)!=FlowerLoRATrainer.base_sha256: raise RuntimeError('Base model file changed')

if __name__=='__main__': main()
