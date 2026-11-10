"""Frozen-weight diagnostics on reviewed train/val crops; no label edits or optimizer steps."""
import json, math
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from common import ROOT,sha
from runtime import setup,CLIP_SHA256
from model_io import load_model
from audit_validation import iou

DATA=ROOT/'train/datasets/flowers_reviewed_v2_trainval'
OUT=ROOT/'train/diagnostics/classification_diagnosis_v1'
ADAPTER=ROOT/'train/runs/flowers_20261104_100817_929606_full/weights/adapter_best.pt'
PROMPTS=['potted peony','potted lily','potted chrysanthemum','potted eucalyptus']

def summary(records,key,split):
 matrix=np.zeros((4,5),dtype=int)
 for r in records:
  if r['split']==split:matrix[r['class_id'],r[key]]+=1
 stats=[]
 for c in range(4):
  tp=int(matrix[c,c]);total=int(matrix[c].sum());pred=int(matrix[:,c].sum())
  stats.append({'class_id':c,'n':total,'precision':tp/pred if pred else 0,'recall':tp/total if total else 0})
 return {'rows':'true class','columns':'predicted 0..3, 4=no detection','matrix':matrix.tolist(),'per_class':stats}

def main():
 OUT.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);setup()
 import clip
 rows=json.loads((DATA/'manifest.json').read_text())
 weights=ROOT/'train/.cache/weights/clip/ViT-B-32.pt';assert sha(weights)==CLIP_SHA256
 payload=torch.load(ADAPTER,map_location='cpu',weights_only=True)
 names=payload['names'];names=[names[k] for k in sorted(names)] if isinstance(names,dict) else names
 assert names==PROMPTS
 norms=[]
 for name,a in payload['state_dict'].items():
  if name.endswith('lora_A'):
   b=payload['state_dict'][name[:-6]+'lora_B'];delta=b.float()@a.float()
   norms.append({'module':name[:-7],'A_norm':a.float().norm().item(),'B_norm':b.float().norm().item(),'delta_norm_unscaled':delta.norm().item(),'finite':bool(torch.isfinite(delta).all())})
 model,pre=clip.load(str(weights),device='cpu',jit=False);model.eval()
 records=[];crops=[]
 for row in rows:
  assert sha(DATA/row['image'])==row['image_sha256']
  with Image.open(DATA/row['image']) as image:
   image=image.convert('RGB')
   for j,o in enumerate(row['objects']):
    xy=o['bbox_xyxy'];crop=image.crop((math.floor(xy[0]),math.floor(xy[1]),math.ceil(xy[2]),math.ceil(xy[3])))
    square=Image.new('RGB',(max(crop.size),)*2,(127,127,127));square.paste(crop,((square.width-crop.width)//2,(square.height-crop.height)//2))
    records.append({'id':row['id'],'object_index':j,'class_id':o['class_id'],'split':row['split'],'bbox_xyxy':xy,'image_sha256':row['image_sha256']})
    crops.append(square)
 with torch.inference_mode():
  text=model.encode_text(clip.tokenize(PROMPTS)).float();text/=text.norm(dim=-1,keepdim=True)
  cache=torch.load(DATA/'images/text_embeddings_clip_ViT-B_32.pt',map_location='cpu',weights_only=True)
  assert set(cache)==set(PROMPTS)
  cache_diffs={p:float((cache[p].float().flatten()-text[i]).abs().max()) for i,p in enumerate(PROMPTS)}
  for start in range(0,len(crops),16):
   feat=model.encode_image(torch.stack([pre(im) for im in crops[start:start+16]])).float();feat/=feat.norm(dim=-1,keepdim=True)
   scores=feat@text.T
   for r,s in zip(records[start:start+16],scores.tolist()):r.update(clip_scores=s,clip_top1=int(np.argmax(s)))
 print('CLIP crops complete',len(records),flush=True);del model
 # GT crops give the detector the target location; this is an oracle diagnostic, not end-to-end evaluation.
 for tag,adapter in [('base',None),('lora',ADAPTER)]:
  model=load_model(ROOT/'models/yolov8s-worldv2.pt',adapter);model.set_classes(PROMPTS)
  for r,crop in zip(records,crops):
   if r['split']!='val':continue
   result=model.predict(np.array(crop)[:,:,::-1].copy(),device='cpu',imgsz=640,conf=.001,iou=.7,verbose=False)[0]
   # Only candidates covering the known whole-pot crop; tiny subobject predictions excluded.
   x1,y1,x2,y2=r['bbox_xyxy'];w=math.ceil(x2)-math.floor(x1);h=math.ceil(y2)-math.floor(y1);side=max(w,h)
   box=[(side-w)//2,(side-h)//2,(side-w)//2+w,(side-h)//2+h]
   pred=[b for b in result.boxes.data.cpu().tolist() if iou(b[:4],box)>=.5]
   scores=[max([b[4] for b in pred if int(b[5])==c],default=0) for c in range(4)]
   r[tag+'_crop_scores']=scores;r[tag+'_crop_top1']=int(np.argmax(scores)) if max(scores)>0 else 4
  print(tag,'crop detection complete',flush=True);del model
 cached=json.loads((ROOT/'train/diagnostics/reviewed_v2_trained_evaluation/reviewed_labels_predictions.json').read_text())
 loc=[{'class_id':c,'gt':0,'localized_any_class':0,'localized_correct_class':0} for c in range(4)]
 for r in cached:
  for g in r['gt']:
   s=loc[g['class']];s['gt']+=1
   candidates=[p for p in r['predictions'] if iou(g['xyxy'],p['xyxy'])>=.5]
   s['localized_any_class']+=bool(candidates);s['localized_correct_class']+=any(p['class']==g['class'] for p in candidates)
 for s in loc:s['any_class_coverage']=s['localized_any_class']/s['gt']
 report={'scope':'Frozen diagnostics; GT crops are oracle localization, not detector acceptance. No optimizer or dataset mutation. No test data used.',
 'adapter_sha256':sha(ADAPTER),'clip_sha256':sha(weights),'prompts':PROMPTS,'adapter_names_match':names==PROMPTS,
 'text_cache_max_abs_diff':cache_diffs,'lora_modules':norms,'clip_train_crops':summary(records,'clip_top1','train'),
 'clip_val_crops':summary(records,'clip_top1','val'),'base_val_crops':summary(records,'base_crop_top1','val'),
 'lora_val_crops':summary(records,'lora_crop_top1','val'),'full_image_coverage_conf025':loc}
 (OUT/'records.json').write_text(json.dumps(records,ensure_ascii=False,indent=2));(OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
 print(json.dumps({k:v for k,v in report.items() if k!='lora_modules'},indent=2),flush=True)
if __name__=='__main__':main()
