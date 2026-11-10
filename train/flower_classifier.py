"""Frozen CLIP visual encoder + supervised linear four-pot classifier.
Train/val separation, train-only augmentation, candidate filtering, class-agnostic NMS.
"""
import json, math, random, argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from common import ROOT,sha
from runtime import setup,CLIP_SHA256
from model_io import load_model
from audit_validation import match,iou

DATA=ROOT/'train/datasets/flowers_reviewed_v2_trainval'
DEFAULT_OUT=ROOT/'train/runs/flower_classifier_v1'
ADAPTER=ROOT/'train/runs/flowers_20261104_100817_929606_full/weights/adapter_best.pt'
PROMPTS=['potted peony','potted lily','potted chrysanthemum','potted eucalyptus']
DISPLAY=['peony','pink lily (original orchid)','yellow flower (original sunflower)','eucalyptus']

def encoder():
 setup()
 import clip
 p=ROOT/'train/.cache/weights/clip/ViT-B-32.pt';assert sha(p)==CLIP_SHA256
 model,pre=clip.load(str(p),device='cpu',jit=False);return model.eval(),pre

def crop_image(im,box,pad=.0):
 x1,y1,x2,y2=box;dx=(x2-x1)*pad;dy=(y2-y1)*pad
 b=(max(0,math.floor(x1-dx)),max(0,math.floor(y1-dy)),min(im.width,math.ceil(x2+dx)),min(im.height,math.ceil(y2+dy)))
 crop=im.crop(b);side=max(crop.size);square=Image.new('RGB',(side,side),(127,127,127));square.paste(crop,((side-crop.width)//2,(side-crop.height)//2));return square

def features(model,pre,images):
 out=[]
 with torch.inference_mode():
  for i in range(0,len(images),16):
   f=model.encode_image(torch.stack([pre(im) for im in images[i:i+16]])).float();out.append(f/f.norm(dim=-1,keepdim=True))
 return torch.cat(out) if out else torch.empty((0,512))

def class_metrics(y,p):
 result=[]
 for c in range(4):
  tp=int(((y==c)&(p==c)).sum());fp=int(((y!=c)&(p==c)).sum());fn=int(((y==c)&(p!=c)).sum())
  result.append({'class_id':c,'tp':tp,'fp':fp,'fn':fn,'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn)})
 return result

def macrof1(stats):return sum(2*s['tp']/max(1,2*s['tp']+s['fp']+s['fn']) for s in stats)/4

def filter_candidates(pred,det_conf,cls_conf,nms_iou=.5):
 def threshold(value,c):
  if isinstance(value,(list,tuple)):
   if len(value)!=4:raise ValueError('Expected four per-class thresholds')
   return value[c]
  return value
 candidates=[p for p in pred if p['class']<4 and p['detector_confidence']>=threshold(det_conf,p['class']) and p['classifier_confidence']>=threshold(cls_conf,p['class'])]
 keep=[]
 for p in sorted(candidates,key=lambda p:-p['confidence']):
  if all(iou(p['xyxy'],q['xyxy'])<=nms_iou for q in keep):keep.append(p)
 return keep

def eval_detection(rows,det_conf,cls_conf):
 stats=[{'class_id':c,'tp':0,'fp':0,'fn':0} for c in range(4)]
 for r in rows:
  pred=filter_candidates(r['candidates'],det_conf,cls_conf);m=match(r['gt'],pred)
  for e in m['predictions']:stats[pred[e['prediction']]['class']]['tp' if e['kind']=='tp' else 'fp']+=1
  for j in m['missed_gt']:stats[r['gt'][j]['class']]['fn']+=1
 for s in stats:s.update(precision=s['tp']/max(1,s['tp']+s['fp']),recall=s['tp']/max(1,s['tp']+s['fn']))
 return stats

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',default=str(DEFAULT_OUT));a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(4)
 rows=json.loads((DATA/'manifest.json').read_text());model,pre=encoder();image_hashes={r['id']:r['image_sha256'] for r in rows}
 feats={};labels={};provenance=[]
 for split in ['train','val']:
  ims=[];ys=[]
  for r in rows:
   if r['split']!=split:continue
   assert r['status']=='accepted' and sha(DATA/r['image'])==r['image_sha256']
   with Image.open(DATA/r['image']) as img:
    img=img.convert('RGB')
    for j,o in enumerate(r['objects']):
     for variant in range(4 if split=='train' else 1):
      padding=0 if variant==0 else random.uniform(.0,.12)
      crop=crop_image(img,o['bbox_xyxy'],padding)
      if variant:
       if random.random()<.5:crop=ImageOps.mirror(crop)
       crop=ImageEnhance.Brightness(crop).enhance(random.uniform(.8,1.2));crop=ImageEnhance.Contrast(crop).enhance(random.uniform(.9,1.1));crop=ImageEnhance.Color(crop).enhance(random.uniform(.85,1.15))
      ims.append(crop);ys.append(o['class_id']);provenance.append({'id':r['id'],'object_index':j,'split':split,'variant':variant,'padding':padding})
  feats[split]=features(model,pre,ims);labels[split]=torch.tensor(ys);print('Encoded',split,len(ys),flush=True)
 torch.save({'features':feats,'labels':labels,'provenance':provenance,'source_manifest_sha256':sha(DATA/'manifest.json')},out/'features.pt')
 head=torch.nn.Linear(512,4);opt=torch.optim.AdamW(head.parameters(),lr=.02,weight_decay=.01)
 weights=1/torch.bincount(labels['train']).float();weights/=weights.mean()
 best=-1;bestloss=float('inf');history=[];beststate=None;bestepoch=0
 for epoch in range(1,301):
  head.train();opt.zero_grad();loss=torch.nn.functional.cross_entropy(head(feats['train']),labels['train'],weight=weights);loss.backward();opt.step()
  with torch.no_grad():
   head.eval();v=head(feats['val']);vl=float(torch.nn.functional.cross_entropy(v,labels['val']));stats=class_metrics(labels['val'],v.argmax(1));score=macrof1(stats)
  history.append({'epoch':epoch,'train_loss':float(loss.detach()),'val_loss':vl,'val_macro_f1':score})
  if score>best or (score==best and vl<bestloss):best=score;bestloss=vl;beststate={k:t.detach().clone() for k,t in head.state_dict().items()};bestepoch=epoch
 head.load_state_dict(beststate);head.eval()
 payload={'format_version':1,'head':beststate,'encoder_sha256':CLIP_SHA256,'detector_adapter_sha256':sha(ADAPTER),'prompts':PROMPTS,'seed':42,'best_epoch':bestepoch,'source_manifest_sha256':sha(DATA/'manifest.json'),'crop_policy':'square gray padding, exact candidate box at inference','training_instances':473,'training_variants':1892}
 torch.save(payload,out/'classifier.pt');(out/'history.json').write_text(json.dumps(history,indent=2))
 with torch.no_grad():crop_stats=class_metrics(labels['val'],head(feats['val']).argmax(1))
 print('GT crop validation',crop_stats,flush=True)
 detector=load_model(ROOT/'models/yolov8s-worldv2.pt',ADAPTER);detector.set_classes(PROMPTS)
 detections=[]
 for r in rows:
  if r['split']!='val':continue
  result=detector.predict(str(DATA/r['image']),device='cpu',imgsz=640,conf=.05,iou=.7,verbose=False)[0]
  boxes=result.boxes.data.cpu().tolist()
  with Image.open(DATA/r['image']) as img:
   img=img.convert('RGB');f=features(model,pre,[crop_image(img,b[:4]) for b in boxes])
  with torch.no_grad():prob=head(f).softmax(1).tolist()
  candidates=[]
  for b,scores in zip(boxes,prob):
   c=int(np.argmax(scores));cp=scores[c]
   candidates.append({'class':c,'xyxy':b[:4],'confidence':float(b[4]*cp),'detector_confidence':b[4],'classifier_confidence':cp,'original_class':int(b[5]),'class_probabilities':scores})
  detections.append({'id':r['id'],'image':r['image'],'image_sha256':r['image_sha256'],'gt':[{'class':o['class_id'],'xyxy':o['bbox_xyxy']} for o in r['objects']],'candidates':candidates})
 print('Candidate classification complete',flush=True)
 grid=[]
 for d in [.05,.1,.15,.25]:
  for c in [.0,.5,.7,.85,.95]:
   stats=eval_detection(detections,d,c);grid.append({'detector_conf':d,'classifier_conf':c,'nms_iou':.5,'stats':stats,'macro_f1':macrof1(stats),'min_precision_recall':min(min(s['precision'],s['recall']) for s in stats),'passes':all(s['precision']>.9 and s['recall']>.9 for s in stats)})
 chosen=max(grid,key=lambda x:(x['passes'],x['min_precision_recall'],x['macro_f1']))
 (out/'val_candidates.json').write_text(json.dumps(detections,indent=2));(out/'threshold_grid.json').write_text(json.dumps(grid,indent=2));(out/'operating_point.json').write_text(json.dumps(chosen,indent=2))
 report={'status':'completed','encoder_frozen':True,'head_train_only':True,'test_used':False,'gt_crop_val':crop_stats,'chosen_val_operating_point':chosen,'selection_note':'Head epoch and thresholds selected on val. These are development metrics, not final held-out test acceptance. No background class trained; rejection relies on detector and classifier thresholds.','best_epoch':bestepoch,'classifier_sha256':sha(out/'classifier.pt'),'source_manifest_sha256':sha(DATA/'manifest.json')}
 (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
