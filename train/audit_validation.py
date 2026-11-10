"""Read-only source audit: paired CPU inference, class-aware matching, review pages."""
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from common import ROOT, sha
from model_io import load_model

OUT=ROOT/'train/diagnostics/validation_audit_v1'
DATA=ROOT/'train/datasets/flowers_video_holdout'
NAMES=['peony','orchid','sunflower','eucalyptus']
PROMPTS=['potted '+x for x in NAMES]
ADAPTERS={'previous':ROOT/'train/diagnostics/adapter_best_architecture_v2.pt',
          'augmented':ROOT/'train/runs/flowers_20261103_200348_146656_full/weights/adapter_best.pt'}

def iou(a,b):
    a=np.array(a);b=np.array(b)
    wh=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]))
    inter=np.prod(wh)
    return float(inter/(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-inter+1e-9))

def match(gt,preds):
    # Greedy confidence-ranked, class-aware, one-to-one matching at IoU .5.
    used=set();events=[]
    for k,p in sorted(enumerate(preds),key=lambda x:-x[1]['confidence']):
        overlaps=[iou(p['xyxy'],g['xyxy']) for g in gt]
        candidates=[j for j,g in enumerate(gt) if j not in used and g['class']==p['class'] and overlaps[j]>=.5]
        if candidates:
            j=max(candidates,key=lambda j:overlaps[j]);used.add(j);kind='tp'
        else:
            j=int(np.argmax(overlaps)) if overlaps else None
            overlap=overlaps[j] if j is not None else 0
            if overlap>=.5:
                kind='duplicate' if gt[j]['class']==p['class'] else 'wrong_class'
            elif overlap>=.1: kind='localization_or_partial'
            else: kind='background_or_unlabeled'
        events.append({'prediction':k,'kind':kind,'nearest_gt':j,'iou':overlaps[j] if j is not None else 0})
    return {'predictions':events,'missed_gt':[j for j in range(len(gt)) if j not in used]}

def panel(img,boxes,title):
    h,w=img.shape[:2];scale=600/w
    out=cv2.resize(img,(600,round(h*scale)))
    colors=[(30,220,255),(255,120,255),(0,200,255),(80,240,80)]
    for k,b in enumerate(boxes):
        x1,y1,x2,y2=[round(x*scale) for x in b['xyxy']]
        color=colors[b['class']]
        cv2.rectangle(out,(x1,y1),(x2,y2),color,2)
        label=f"{k}:{NAMES[b['class']]}"+ (f" {b['confidence']:.2f}" if 'confidence' in b else '')
        y=max(14,y1)
        cv2.putText(out,label,(max(0,x1),y),cv2.FONT_HERSHEY_SIMPLEX,.43,(0,0,0),3)
        cv2.putText(out,label,(max(0,x1),y),cv2.FONT_HERSHEY_SIMPLEX,.43,color,1)
    out=cv2.copyMakeBorder(out,28,0,0,0,cv2.BORDER_CONSTANT)
    cv2.putText(out,title,(6,19),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
    return out

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    files=sorted((DATA/'images/val').glob('*.png'))
    rows=[]
    for i,p in enumerate(files):
        img=cv2.imread(str(p));h,w=img.shape[:2];label=DATA/'labels/val'/p.with_suffix('.txt').name
        gt=[]
        for line in label.read_text().splitlines():
            c,x,y,bw,bh=map(float,line.split())
            gt.append({'class':int(c),'xyxy':[(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]})
        rows.append({'id':i+1,'image':str(p),'image_sha256':sha(p),'label_sha256':sha(label),'gt':gt,'models':{}})
    for tag,path in ADAPTERS.items():
        model=load_model(ROOT/'models/yolov8s-worldv2.pt',path);model.set_classes(PROMPTS)
        for row in rows:
            r=model.predict(row['image'],device='cpu',imgsz=640,conf=.25,iou=.7,verbose=False)[0]
            pred=[{'class':int(b[5]),'confidence':float(b[4]),'xyxy':b[:4]} for b in r.boxes.data.cpu().tolist()]
            row['models'][tag]={'boxes':pred,'matching':match(row['gt'],pred)}
        del model
        print(tag,'done',flush=True)
    pages=[]
    for start in range(0,len(rows),3):
        tiles=[]
        for row in rows[start:start+3]:
            img=cv2.imread(row['image']);title=f"{row['id']:02d} {Path(row['image']).stem}"
            panels=[panel(img,row['gt'],title+' | GT')]
            panels += [panel(img,row['models'][tag]['boxes'],tag+' | conf >= .25') for tag in ADAPTERS]
            tile=np.hstack(panels);tiles.append(tile)
            cv2.imwrite(str(OUT/f"image_{row['id']:02d}.jpg"),tile)
        page=f'page_{start//3+1:02d}.jpg';cv2.imwrite(str(OUT/page),np.vstack(tiles));pages.append(page)
    report={'conf':.25,'nms_iou':.7,'match_iou':.5,'device':'cpu','imgsz':640,
            'adapters':{k:{'path':str(v),'sha256':sha(v)} for k,v in ADAPTERS.items()},'images':rows}
    (OUT/'predictions.json').write_text(json.dumps(report,indent=2))
    html='<meta charset="utf-8"><h1>39张验证图：原标签 / 上轮 / 增强轮</h1><p>固定conf=.25，匹配IoU=.5；自动匹配不是人工真值。</p>'
    html+=''.join(f'<h2>{r["id"]:02d} {Path(r["image"]).stem}</h2><a href="{r["image"]}">原图</a><img style="width:100%" src="image_{r["id"]:02d}.jpg">' for r in rows)
    (OUT/'review.html').write_text(html)
    print('ready',OUT,flush=True)

if __name__=='__main__':main()
