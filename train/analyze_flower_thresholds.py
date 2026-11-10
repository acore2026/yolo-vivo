"""Choose class-specific thresholds on validation only; retain raw predictions."""
import json
from pathlib import Path
import numpy as np
import torch
import cv2
from common import ROOT,config,sha
from model_io import load_model
from audit_validation import match


def main():
    torch.set_num_threads(2)
    out=ROOT/'train/diagnostics/clip_thresholds_v1';out.mkdir(parents=True,exist_ok=False)
    cfg=config('train/configs/flowers_clip_lily_chrysanthemum.yaml');data=config(cfg['data'])
    root=Path(data['path']);adapter=ROOT/'train/runs/flowers_20261103_210003_062186_full/weights/adapter_best.pt'
    model=load_model(ROOT/cfg['model'],adapter);model.set_classes(cfg['prompts'])
    records=[]
    for path in sorted((root/'images/val').glob('*.png')):
        img=cv2.imread(str(path));h,w=img.shape[:2];gt=[]
        for line in (root/'labels/val'/path.with_suffix('.txt').name).read_text().splitlines():
            c,x,y,bw,bh=map(float,line.split());gt.append({'class':int(c),'xyxy':[(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]})
        r=model.predict(img,device='cpu',imgsz=640,conf=.001,iou=.7,verbose=False)[0]
        pred=[{'class':int(b[5]),'confidence':float(b[4]),'xyxy':b[:4]} for b in r.boxes.data.tolist()]
        matching=match(gt,pred);tp={e['prediction'] for e in matching['predictions'] if e['kind']=='tp'}
        for i,p in enumerate(pred):p['tp']=i in tp
        records.append({'image':str(path),'gt':gt,'predictions':pred})
    results=[]
    for c in range(4):
        ng=sum(g['class']==c for r in records for g in r['gt'])
        predictions=[p for r in records for p in r['predictions'] if p['class']==c]
        thresholds=sorted({.25,*[p['confidence'] for p in predictions]})
        curve=[]
        for t in thresholds:
            selected=[p for p in predictions if p['confidence']>=t];n=len(selected);tp=sum(p['tp'] for p in selected)
            precision=tp/n if n else 0;recall=tp/ng
            curve.append({'threshold':t,'tp':tp,'fp':n-tp,'fn':ng-tp,'precision':precision,'recall':recall,
                          'f1':2*precision*recall/(precision+recall) if precision+recall else 0})
        eligible=[r for r in curve if r['precision']>=.9 and r['recall']>=.9]
        results.append({'class_id':c,'prompt':cfg['prompts'][c],'gt_count':ng,
            'at_025':next(r for r in curve if r['threshold']==.25),
            'best_f1':max(curve,key=lambda r:(r['f1'],r['threshold'])),
            'both_90_feasible':bool(eligible),'both_90_best_f1':max(eligible,key=lambda r:r['f1']) if eligible else None,
            'curve':curve})
    report={'scope':'val-only threshold fitting; no independent acceptance claim','adapter_sha256':sha(adapter),
            'match_iou':.5,'nms_iou':.7,'minimum_prediction_conf':.001,'results':results}
    (out/'report.json').write_text(json.dumps(report,indent=2))
    (out/'predictions.json').write_text(json.dumps(records,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(10,7))
    for ax,r in zip(axs.flat,results):
        for key in ('precision','recall'):ax.plot([v['threshold'] for v in r['curve']],[v[key] for v in r['curve']],label=key)
        ax.axhline(.9,color='gray',linestyle='--');ax.set(title=r['prompt'],xlabel='Confidence threshold',ylim=(0,1.05));ax.legend()
    fig.tight_layout();fig.savefig(out/'threshold_curves.png',dpi=160)
    print(json.dumps([{k:v for k,v in r.items() if k!='curve'} for r in results],indent=2))

if __name__=='__main__':main()
