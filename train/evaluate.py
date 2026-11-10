"""Evaluate base versus adapter on the SAME labeled dataset and prompts."""
import argparse
import json
import yaml
from common import resolve, sha

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',default='models/yolov8s-worldv2.pt')
    p.add_argument('--adapter',required=True)
    p.add_argument('--data',required=True,help='Flower or old-category labeled evaluation YAML')
    p.add_argument('--split',choices=['val','test'],default='test')
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='0')
    a=p.parse_args();out=resolve(a.output)
    if out.exists(): raise SystemExit('Output directory already exists')
    data=resolve(a.data);d=yaml.safe_load(data.read_text());names=d['names']
    names=[names[k] for k in sorted(names)] if isinstance(names,dict) else names
    if a.split not in d: raise SystemExit('Requested evaluation split missing')
    from model_io import load_model
    out.mkdir(parents=True)
    report={'base_sha256':sha(resolve(a.base)),'adapter_sha256':sha(resolve(a.adapter)),'prompts':names,'split':a.split,'metrics':{}}
    for tag,adapter in [('base',None),('lora',resolve(a.adapter))]:
        model=load_model(resolve(a.base),adapter);model.set_classes(names)
        result=model.val(data=str(data),split=a.split,imgsz=640,batch=8,device=a.device,conf=0.001,iou=0.7,project=str(out),name=tag)
        report['metrics'][tag]={k:float(v) for k,v in result.results_dict.items()}
        report.setdefault('per_class', {})[tag] = [
            {'class_id':int(cls), 'name':names[int(cls)],
             'precision':float(result.box.p[i]), 'recall':float(result.box.r[i]),
             'ap50':float(result.box.ap50[i]), 'ap50_95':float(result.box.ap[i])}
            for i,cls in enumerate(result.box.ap_class_index)]
        del model
    report['delta']={k:report['metrics']['lora'][k]-v for k,v in report['metrics']['base'].items()}
    (out/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
