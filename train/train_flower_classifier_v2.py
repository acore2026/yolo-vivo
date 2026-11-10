"""Train-only reviewed rejection samples + mild side truncation, frozen CLIP."""
import json,random,argparse
import numpy as np
import torch
from PIL import Image,ImageOps,ImageEnhance
from flower_classifier import *
OUT=ROOT/'train/runs/flower_classifier_v2'
REVIEW=ROOT/'train/diagnostics/flower_hard_review_v2'

def main():
 global OUT,REVIEW
 parser=argparse.ArgumentParser();parser.add_argument('--output',default=str(OUT));parser.add_argument('--reject-weight',type=float,default=1.);parser.add_argument('--reuse-extra',default=None);parser.add_argument('--review-dir',default=str(REVIEW));args=parser.parse_args();OUT=Path(args.output);REVIEW=Path(args.review_dir)
 assert 0<args.reject_weight<=1
 OUT.mkdir(exist_ok=False);torch.set_num_threads(4);torch.manual_seed(42);random.seed(42)
 old=torch.load(DEFAULT_OUT/'features.pt',map_location='cpu',weights_only=True)
 assert old['source_manifest_sha256']==sha(DATA/'manifest.json')
 enc,pre=encoder();ims=[];ys=[];prov=[]
 reviewed=json.loads((REVIEW/'reviewed.json').read_text())
 for r in reviewed:
  assert r['status']=='accepted' and r['split']=='train' and r['reviewer']=='assistant_visual'
  assert sha(REVIEW/r['crop'])==r['crop_sha256']
  with Image.open(REVIEW/r['crop']) as im:
   im=im.convert('RGB')
   for v in range(4):
    crop=im.copy()
    if v%2:crop=ImageOps.mirror(crop)
    if v>1:crop=ImageEnhance.Brightness(crop).enhance(.8 if v==2 else 1.2)
    ims.append(crop_image(crop,[0,0,crop.width,crop.height]));ys.append(r['class_id']);prov.append({'review_id':r['id'],'source_id':r['source_id'],'variant':v,'class_id':r['class_id'],'split':'train'})
 # Mild left/right truncation only, retain 82% width and all vertical extent.
 for r in json.loads((DATA/'manifest.json').read_text()):
  if r['split']!='train':continue
  with Image.open(DATA/r['image']) as im:
   im=im.convert('RGB')
   for j,o in enumerate(r['objects']):
    x1,y1,x2,y2=o['bbox_xyxy'];dx=(x2-x1)*.18
    for side,box in [('left',[x1+dx,y1,x2,y2]),('right',[x1,y1,x2-dx,y2])]:
     ims.append(crop_image(im,box));ys.append(o['class_id']);prov.append({'source_id':r['id'],'object_index':j,'side':side,'retained_width':.82,'bbox_xyxy':box,'split':'train','scope':'augmentation derived from reviewed positive; 16 sampled previews reviewed, not every transformed image'})
 if args.reuse_extra:
  cached=torch.load(args.reuse_extra,map_location='cpu',weights_only=True)
  assert cached['original_features_sha256']==sha(DEFAULT_OUT/'features.pt')
  assert cached['provenance']==prov and torch.equal(cached['extra_labels'],torch.tensor(ys))
  extra=cached['extra_features']
 else:extra=features(enc,pre,ims)
 assert len(extra)==len(ys) and torch.isfinite(extra).all()
 x=torch.cat([old['features']['train'],extra]);y=torch.cat([old['labels']['train'],torch.tensor(ys)])
 torch.save({'extra_features':extra,'extra_labels':torch.tensor(ys),'provenance':prov,'original_features_sha256':sha(DEFAULT_OUT/'features.pt')},OUT/'extra_features.pt')
 print('Train examples',len(y),'class counts',torch.bincount(y).tolist(),flush=True)
 head=torch.nn.Linear(512,5);opt=torch.optim.AdamW(head.parameters(),lr=.02,weight_decay=.01);weights=1/torch.bincount(y).float();weights/=weights.mean();weights[4]*=args.reject_weight;best=(-1,float('-inf'));history=[]
 for epoch in range(1,301):
  opt.zero_grad();loss=torch.nn.functional.cross_entropy(head(x),y,weight=weights);loss.backward();opt.step()
  with torch.no_grad():
   logits=head(old['features']['val']);vl=float(torch.nn.functional.cross_entropy(logits,old['labels']['val']));stats=class_metrics(old['labels']['val'],logits.argmax(1));score=macrof1(stats)
  history.append({'epoch':epoch,'train_loss':float(loss.detach()),'val_loss':vl,'val_macro_f1':score})
  if (score,-vl)>best:best=(score,-vl);state={k:v.detach().clone() for k,v in head.state_dict().items()};bestepoch=epoch
 head.load_state_dict(state);head.eval()
 with torch.no_grad():crop_stats=class_metrics(old['labels']['val'],head(old['features']['val']).argmax(1))
 payload={'format_version':2,'head':state,'encoder_sha256':CLIP_SHA256,'detector_adapter_sha256':sha(ADAPTER),'prompts':PROMPTS,'reject_class':4,'seed':42,'best_epoch':bestepoch,'source_manifest_sha256':sha(DATA/'manifest.json'),'review_manifest_sha256':sha(REVIEW/'reviewed.json'),'reject_loss_multiplier':args.reject_weight,'training_counts':torch.bincount(y).tolist()}
 torch.save(payload,OUT/'classifier.pt');(OUT/'history.json').write_text(json.dumps(history,indent=2))
 # Reuse detector proposals unchanged, recompute classification with five-way head.
 records=json.loads((DEFAULT_OUT/'candidate_recovery/val_candidates.json').read_text())
 for r in records:
  with Image.open(DATA/r['image']) as im:f=features(enc,pre,[crop_image(im.convert('RGB'),b['xyxy']) for b in r['candidates']])
  with torch.no_grad():prob=head(f).softmax(1).tolist()
  for b,s in zip(r['candidates'],prob):
   c=int(np.argmax(s));b.update({'class':c,'classifier_confidence':s[c],'confidence':b['detector_confidence']*s[c],'class_probabilities':s})
 grid=[]
 for d in [.01,.025,.05,.1,.15,.25]:
  for c in [0,.3,.5,.6,.7,.85,.95]:
   stats=eval_detection(records,d,c);grid.append({'detector_conf':d,'classifier_conf':c,'nms_iou':.5,'stats':stats,'macro_f1':macrof1(stats),'min_precision_recall':min(min(s['precision'],s['recall']) for s in stats),'passes':all(s['precision']>.9 and s['recall']>.9 for s in stats)})
 chosen=max(grid,key=lambda x:(x['passes'],x['min_precision_recall'],x['macro_f1']))
 for name,value in [('val_candidates.json',records),('threshold_grid.json',grid),('operating_point.json',chosen)]: (OUT/name).write_text(json.dumps(value,indent=2))
 report={'gt_crop_val':crop_stats,'chosen_val_operating_point':chosen,'best_epoch':bestepoch,'classifier_sha256':sha(OUT/'classifier.pt'),'test_used':False,'train_counts':torch.bincount(y).tolist(),'reviewed_crops':len(reviewed),'reviewed_reject_crops':sum(r['class_id']==4 for r in reviewed),'scope':'val-selected development metrics; reject head class4; no new YOLO labels'}
 (OUT/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
