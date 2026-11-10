"""Three flower classes plus internal background rejection; frozen CLIP encoder.
Train/calibrate uses train/val only. Test inference never changes parameters.
"""
import argparse,json,random
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageOps,ImageEnhance
from common import ROOT,resolve,sha
from three_flower_data import PROMPTS,NAMES,verify,dump
from flower_classifier import encoder,features,crop_image
from model_io import load_model
from runtime import CLIP_SHA256
from audit_validation import match,iou

def keep(candidates,d,c):
 result=[]
 for b in sorted(candidates,key=lambda b:-b['confidence']):
  if b['class']<3 and b['detector_confidence']>=d and b['classifier_confidence']>=c and all(iou(b['xyxy'],x['xyxy'])<=.5 for x in result):result.append(b)
 return result

def stats(records,d,c):
 result=[dict(class_id=k,tp=0,fp=0,fn=0) for k in range(3)]
 for r in records:
  p=keep(r['candidates'],d,c);m=match(r['gt'],p)
  for event in m['predictions']:result[p[event['prediction']]['class']]['tp' if event['kind']=='tp' else 'fp']+=1
  for j in m['missed_gt']:result[r['gt'][j]['class']]['fn']+=1
 for s in result:s.update(precision=s['tp']/max(1,s['tp']+s['fp']),recall=s['tp']/max(1,s['tp']+s['fn']))
 return result

def candidates(det,enc,pre,head,im,device):
 b=det.predict(np.asarray(im)[:,:,::-1].copy(),device=device,imgsz=640,conf=.01,iou=.7,verbose=False)[0].boxes.data.cpu().tolist();f=features(enc,pre,[crop_image(im,x[:4]) for x in b])
 with torch.inference_mode():scores=head(f).softmax(1).tolist()
 result=[]
 for box,s in zip(b,scores):
  cls=int(np.argmax(s));result.append({'class':cls,'xyxy':box[:4],'confidence':box[4]*s[cls],'detector_confidence':box[4],'classifier_confidence':s[cls]})
 return result

def load(folder,device):
 payload=torch.load(folder/'classifier.pt',map_location='cpu',weights_only=True)
 assert payload['prompts']==PROMPTS and payload['reject_class']==3 and payload['encoder_sha256']==CLIP_SHA256
 adapter=Path(payload['adapter'])
 # Checkpoints historically embed the source computer's absolute path.
 if adapter.is_absolute() and 'train/runs/' in adapter.as_posix():
  adapter=ROOT/'train/runs'/adapter.as_posix().split('train/runs/',1)[1]
 elif not adapter.is_absolute():
  adapter=ROOT/adapter
 assert sha(adapter)==payload['detector_adapter_sha256']
 head=torch.nn.Linear(512,4);head.load_state_dict(payload['head']);head.eval();enc,pre=encoder()
 det=load_model(ROOT/'models/yolov8s-worldv2.pt',adapter);det.set_classes(PROMPTS)
 return det,enc,pre,head

def train(a):
 data=resolve(a.data);rows,report=verify(data);out=resolve(a.output);out.mkdir(parents=True,exist_ok=False);adapter=resolve(a.adapter)
 payload=torch.load(adapter,map_location='cpu',weights_only=True);assert list(payload['names'].values())==PROMPTS if isinstance(payload['names'],dict) else payload['names']==PROMPTS
 torch.manual_seed(42);random.seed(42);enc,pre=encoder();xs={};ys={};provenance=[]
 for split in ['train','val']:
  images=[];labels=[]
  for r in rows:
   if r['split']!=split:continue
   with Image.open(data/r['image']) as raw:
    raw=raw.convert('RGB');boxes=[(b['class_id'],b['bbox_xyxy']) for b in r['objects']]
    if not boxes:boxes=[(3,[0,0,raw.width,raw.height])]
    for j,(cls,box) in enumerate(boxes):
     for variant in range(4 if split=='train' else 1):
      im=crop_image(raw,box)
      if variant%2:im=ImageOps.mirror(im)
      if variant>=2:im=ImageEnhance.Brightness(im).enhance(.8 if variant==2 else 1.2)
      images.append(im);labels.append(cls);provenance.append(dict(id=r['id'],object_index=j,split=split,variant=variant,class_id=cls))
  # Reuse explicitly reviewed old difficult crops with source membership/hash checks.
  # Eucalyptus crops are excluded, never silently relabeled as background.
  if split=='train':
   review=ROOT/'train/diagnostics/flower_combined_review_v3';byid={r['id']:r for r in rows if r['split']=='train'}
   for r in json.loads((review/'reviewed.json').read_text()):
    if r['class_id']==3 or r['source_id'] not in byid:continue
    assert r['status']=='accepted' and r['reviewer']=='assistant_visual' and r['split']=='train'
    assert r['source_image_sha256']==byid[r['source_id']]['image_sha256'];assert sha(review/r['crop'])==r['crop_sha256']
    with Image.open(review/r['crop']) as raw:
     im=crop_image(raw.convert('RGB'),[0,0,raw.width,raw.height]);cls=3 if r['class_id']==4 else r['class_id']
     images.append(im);labels.append(cls);provenance.append(dict(id=r['id'],source_id=r['source_id'],split='train',class_id=cls,review_reused=True,review_manifest_sha256=sha(review/'reviewed.json')))
  xs[split]=features(enc,pre,images);ys[split]=torch.tensor(labels);print('Encoded',split,len(labels),flush=True)
 torch.save(dict(features=xs,labels=ys,provenance=provenance,source_manifest_sha256=sha(data/'manifest.json')),out/'features.pt')
 head=torch.nn.Linear(512,4);opt=torch.optim.AdamW(head.parameters(),lr=.02,weight_decay=.01);counts=torch.bincount(ys['train'],minlength=4);assert (counts>0).all();weights=1/counts.float();weights/=weights.mean();weights[3]*=.25;history=[];best=None
 for epoch in range(1,301):
  opt.zero_grad();loss=torch.nn.functional.cross_entropy(head(xs['train']),ys['train'],weight=weights);loss.backward();opt.step()
  with torch.no_grad():
   logits=head(xs['val']);vl=float(torch.nn.functional.cross_entropy(logits,ys['val']));pred=logits.argmax(1);f1=sum(2*int(((pred==k)&(ys['val']==k)).sum())/max(1,int((pred==k).sum()+(ys['val']==k).sum())) for k in range(3))/3
  history.append(dict(epoch=epoch,train_loss=float(loss.detach()),val_loss=vl,val_macro_f1=f1))
  if best is None or (f1,-vl)>best:best=(f1,-vl);state={k:v.detach().clone() for k,v in head.state_dict().items()};bestepoch=epoch
 head.load_state_dict(state);head.eval();torch.save(dict(format_version=3,head=state,prompts=PROMPTS,display_names=NAMES,reject_class=3,encoder_sha256=CLIP_SHA256,adapter=str(adapter),detector_adapter_sha256=sha(adapter),source_manifest_sha256=sha(data/'manifest.json'),training_counts=counts.tolist(),best_epoch=bestepoch),out/'classifier.pt');dump(out/'history.json',history)
 det=load_model(ROOT/'models/yolov8s-worldv2.pt',adapter);det.set_classes(PROMPTS);records=[]
 for r in rows:
  if r['split']=='val':
   with Image.open(data/r['image']) as im:pred=candidates(det,enc,pre,head,im.convert('RGB'),a.device)
   records.append(dict(id=r['id'],gt=[{'class':b['class_id'],'xyxy':b['bbox_xyxy']} for b in r['objects']],candidates=pred))
 grid=[]
 for d in [.01,.025,.05,.1,.15,.25]:
  for c in [0,.3,.5,.6,.7,.85,.95]:
   s=stats(records,d,c);grid.append(dict(detector_conf=d,classifier_conf=c,nms_iou=.5,stats=s,minimum_pr=min(x[k] for x in s for k in ['precision','recall']),macro_f1=sum(2*x['tp']/max(1,2*x['tp']+x['fp']+x['fn']) for x in s)/3))
 point=max(grid,key=lambda t:(t['minimum_pr'],t['macro_f1']));point.update(classifier_sha256=sha(out/'classifier.pt'),passes=point['minimum_pr']>.9,scope='development val only; not independent acceptance')
 dump(out/'val_candidates.json',records);dump(out/'threshold_grid.json',grid);dump(out/'operating_point.json',point);dump(out/'report.json',dict(status='completed',classes=NAMES,internal_reject_class=3,test_used=False,point=point,source_report=report));print(json.dumps(point,indent=2))

def test(a):
 import cv2
 for src in a.sources:
  if not resolve(src).is_file():raise FileNotFoundError(resolve(src))
 folder=resolve(a.classifier_dir);point=json.loads((folder/'operating_point.json').read_text());assert point['classifier_sha256']==sha(folder/'classifier.pt');out=resolve(a.output);out.mkdir(parents=True,exist_ok=False);dump(out/'frozen_operating_point.json',point);det,enc,pre,head=load(folder,a.device);summary=dict(model='three-class YOLO-World LoRA + frozen CLIP classifier',classifier_sha256=sha(folder/'classifier.pt'),operating_point=point,inputs=[])
 for i,src in enumerate(a.sources):
  source=resolve(src);video=source.suffix.lower() in ['.mov','.mp4'];cap=cv2.VideoCapture(str(source)) if video else None;expected=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if video else 1;fps=cap.get(cv2.CAP_PROP_FPS) if video else None;writer=None;n=0;target=out/f'{i+1:02d}_{source.stem}{".mp4" if video else ".jpg"}'
  with (out/f'{i+1:02d}_{source.stem}.jsonl').open('x') as log:
   try:
    while n<expected:
     if video:ok,frame=cap.read();assert ok
     else:frame=cv2.imread(str(source));assert frame is not None
     boxes=keep(candidates(det,enc,pre,head,Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)),a.device),point['detector_conf'],point['classifier_conf']);log.write(json.dumps(dict(frame=n,detections=boxes))+'\n')
     for b in boxes:
      x1,y1,x2,y2=map(round,b['xyxy']);cv2.rectangle(frame,(x1,y1),(x2,y2),(0,220,255),2);cv2.putText(frame,['peony','lily','chrysanthemum'][b['class']]+f' {b["confidence"]:.2f}',(max(0,x1),max(25,y1-8)),0,.8,(0,180,255),2)
     if video:
      if writer is None:writer=cv2.VideoWriter(str(target),cv2.VideoWriter_fourcc(*'mp4v'),fps,(frame.shape[1],frame.shape[0]));assert writer.isOpened()
      writer.write(frame)
     else:assert cv2.imwrite(str(target),frame)
     n+=1
     if n%100==0:print(source.name,n,expected,flush=True)
   finally:
    if cap:cap.release()
    if writer:writer.release()
  summary['inputs'].append(dict(source=str(source),source_sha256=sha(source),frames=n,output=str(target)))
 dump(out/'summary.json',summary)
 if a.test_data:
  data=resolve(a.test_data);rows,_=verify(data);records=[]
  for r in rows:
   with Image.open(data/r['image']) as im:pred=candidates(det,enc,pre,head,im.convert('RGB'),a.device)
   records.append(dict(id=r['id'],gt=[{'class':b['class_id'],'xyxy':b['bbox_xyxy']} for b in r['objects']],candidates=pred))
  dump(out/'flower3_metrics.json',dict(scope='Previously exposed flower3 regression; reused reviewed labels',stats=stats(records,point['detector_conf'],point['classifier_conf']),records=records))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','test']);p.add_argument('--data',default='train/datasets/flowers_three_v1');p.add_argument('--adapter');p.add_argument('--classifier-dir');p.add_argument('--output',required=True);p.add_argument('--device',default='cpu');p.add_argument('--test-data');p.add_argument('--sources',nargs='+',default=['test_videos/test.JPG','test_videos/test.MOV','test_videos/test2.MOV']);a=p.parse_args();torch.set_num_threads(4);train(a) if a.mode=='train' else test(a)
