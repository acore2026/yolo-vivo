"""Versioned visual review workspace; no automatic acceptance."""
import json,hashlib,copy,shutil
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'train/datasets/flowers_clip_lily_chrysanthemum_v1'
OUT=ROOT/'train/diagnostics/label_review_v2'
NAMES=['0 peony','1 lily (orchid)','2 chrys (sunflower)','3 eucalyptus']
COLORS=[(255,140,60),(240,80,240),(0,210,255),(70,230,90)]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def boxes(r):
 im=cv2.imread(str(SRC/r['image']));h,w=im.shape[:2];out=[]
 for line in (SRC/r['label']).read_text().splitlines():
  c,x,y,bh,bv=map(float,line.split());out.append({'class_id':int(c),'bbox_xyxy':[(x-bh/2)*w,(y-bv/2)*h,(x+bh/2)*w,(y+bv/2)*h]})
 return im,out
def tile(r,objects=None):
 im,b=boxes(r);h,w=im.shape[:2];raw=cv2.resize(im,(640,360));ann=raw.copy()
 for i,o in enumerate(b if objects is None else objects):
  c=o['class_id'];xy=[round(v*.5) for v in o['bbox_xyxy']];cv2.rectangle(ann,tuple(xy[:2]),tuple(xy[2:]),COLORS[c],1)
  cv2.putText(ann,f'{i}: '+NAMES[c],(max(0,xy[0]),max(15,xy[1])),cv2.FONT_HERSHEY_SIMPLEX,.43,COLORS[c],1)
 out=np.vstack([np.zeros((28,1280,3),dtype=np.uint8),np.hstack([raw,ann])]);cv2.putText(out,r['id']+' | '+r['split']+' | RAW / LABELS',(8,20),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1);return out
def make_pages(rows,prefix):
 paths=[]
 for start in range(0,len(rows),4):
  name=f'{prefix}_{start//4+1:02d}.jpg';cv2.imwrite(str(OUT/name),np.vstack([tile(r) for r in rows[start:start+4]]));paths.append({'file':name,'ids':[r['id'] for r in rows[start:start+4]]})
 return paths
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 assert not (OUT/'index.json').exists()
 allrows=json.loads((SRC/'manifest.json').read_text());rows=[r for r in allrows if r['split'] in ('train','val')]
 by={r['id']:r for r in rows};train=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='val']
 mixed=[r for r in train if r['id'].startswith('flower1_')]
 pilot=[mixed[i] for i in [0,14,28,42,56,70]]
 for g in ['flower4','flower5','flower6','flower7']:
  arr=[r for r in train if r['id'].startswith(g+'_')];pilot.extend(arr[i] for i in ([10,35] if g in ['flower5','flower6'] else [25]))
 pilot.extend(val[i] for i in [0,5,12,17,23,27,34,38])
 assert len(pilot)==20
 ids={r['id'] for r in pilot};remaining=[r for r in rows if r['id'] not in ids]
 index={'source':str(SRC),'pilot':make_pages(pilot,'pilot'),'remaining':make_pages(remaining,'remaining'),'count':len(rows)}
 save(OUT/'index.json',index);save(OUT/'source_integrity_before.json',{r['id']:{'image':sha(SRC/r['image']),'label':sha(SRC/r['label'])} for r in allrows})
 save(OUT/'decisions.json',{})
 print(json.dumps(index['pilot'],ensure_ascii=False));print('Review count',len(rows),'remaining pages',len(index['remaining']))
if __name__=='__main__':main()
