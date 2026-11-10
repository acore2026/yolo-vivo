"""Validation-only candidate recovery for a frozen classifier. No training/test use."""
import json
from pathlib import Path
import torch,numpy as np
from PIL import Image
from flower_classifier import *

def main():
 torch.set_num_threads(4);out=DEFAULT_OUT/'candidate_recovery';out.mkdir(exist_ok=False)
 enc,pre=encoder();head=torch.nn.Linear(512,4);head.load_state_dict(torch.load(DEFAULT_OUT/'classifier.pt',map_location='cpu',weights_only=True)['head']);head.eval()
 detector=load_model(ROOT/'models/yolov8s-worldv2.pt',ADAPTER);detector.set_classes(PROMPTS)
 rows=json.loads((DATA/'manifest.json').read_text());records=[]
 for row in rows:
  if row['split']!='val':continue
  result=detector.predict(str(DATA/row['image']),device='cpu',imgsz=640,conf=.01,iou=.7,verbose=False)[0]
  boxes=result.boxes.data.cpu().tolist()
  with Image.open(DATA/row['image']) as im:
   f=features(enc,pre,[crop_image(im.convert('RGB'),b[:4]) for b in boxes])
  with torch.no_grad():scores=head(f).softmax(1).tolist()
  pred=[]
  for b,s in zip(boxes,scores):
   c=int(np.argmax(s));pred.append({'class':c,'xyxy':b[:4],'confidence':b[4]*s[c],'detector_confidence':b[4],'classifier_confidence':s[c],'original_class':int(b[5]),'class_probabilities':s})
  records.append({'id':row['id'],'image':row['image'],'image_sha256':row['image_sha256'],'gt':[{'class':o['class_id'],'xyxy':o['bbox_xyxy']} for o in row['objects']],'candidates':pred})
 grid=[]
 for d in [.01,.025,.05,.1,.15,.25]:
  for c in [.0,.3,.5,.6,.7,.85,.95]:
   stats=eval_detection(records,d,c);grid.append({'detector_conf':d,'classifier_conf':c,'nms_iou':.5,'stats':stats,'macro_f1':macrof1(stats),'min_precision_recall':min(min(s['precision'],s['recall']) for s in stats),'passes':all(s['precision']>.9 and s['recall']>.9 for s in stats)})
 chosen=max(grid,key=lambda x:(x['passes'],x['min_precision_recall'],x['macro_f1']))
 for name,value in [('val_candidates.json',records),('threshold_grid.json',grid),('operating_point.json',chosen)]: (out/name).write_text(json.dumps(value,indent=2))
 print(json.dumps(chosen,indent=2),flush=True)
if __name__=='__main__':main()
