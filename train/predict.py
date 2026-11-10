"""Separate base and flower branches; outputs remain separately identified."""
import argparse
import json
from common import resolve, config

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='train/configs/flowers.yaml')
    p.add_argument('--adapter',required=True)
    p.add_argument('--source',required=True,help='Local image/video path (not camera)')
    p.add_argument('--old-prompts',nargs='+',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='0')
    a=p.parse_args();c=config(a.config);source=resolve(a.source);out=resolve(a.output)
    if not source.is_file(): raise SystemExit('Local source file required')
    if out.exists(): raise SystemExit('Output directory already exists')
    from model_io import load_model
    out.mkdir(parents=True)
    for branch,adapter,prompts in [('original',None,a.old_prompts),('flowers',resolve(a.adapter),c['prompts'])]:
        model=load_model(resolve(c['model']),adapter);model.set_classes(prompts)
        with (out/f'{branch}.jsonl').open('w') as f:
            for index,r in enumerate(model.predict(source=str(source),stream=True,device=a.device,conf=0.25,iou=0.7,imgsz=640,save=True,project=str(out),name=branch)):
                f.write(json.dumps({'index':index,'path':r.path,'branch':branch,'detections':json.loads(r.to_json())},ensure_ascii=False)+'\n')
        del model

if __name__=='__main__': main()
