"""One frozen-cascade held-out test evaluation after independent label review. No tuning."""
import json
import torch,numpy as np,cv2
from PIL import Image
from flower_classifier import *

def main():
 torch.set_num_threads(4)
 run=ROOT/'train/runs/flower_classifier_v3';review=ROOT/'train/diagnostics/flower_test_review_v3';data=ROOT/'train/datasets/flowers_test_reviewed_v3';out=run/'frozen_test';out.mkdir(exist_ok=False)
 lock=json.loads((review/'frozen_test_inputs.json').read_text());pointpath=run/'per_class_calibration/operating_point.json'
 assert sha(run/'classifier.pt')==lock['classifier_sha256'];assert sha(pointpath)==lock['operating_point_sha256'];assert sha(ADAPTER)==lock['adapter_sha256'];assert sha(data/'manifest.json')==lock['reviewed_test_manifest_sha256']
 point=json.loads(pointpath.read_text());payload=torch.load(run/'classifier.pt',map_location='cpu',weights_only=True);head=torch.nn.Linear(512,5);head.load_state_dict(payload['head']);head.eval();enc,pre=encoder();det=load_model(ROOT/'models/yolov8s-worldv2.pt',ADAPTER);det.set_classes(PROMPTS)
 records=[];rows=json.loads((data/'manifest.json').read_text());assert len(rows)==80
 for idx,r in enumerate(rows):
  assert r['status']=='accepted' and r['review']['reviewer']=='assistant_visual'
  assert sha(data/r['image'])==r['image_sha256'] and sha(data/r['label'])==r['label_sha256']
  result=det.predict(str(data/r['image']),device='cpu',imgsz=640,conf=min(point['detector_conf']),iou=.7,verbose=False)[0];boxes=result.boxes.data.cpu().tolist()
  with Image.open(data/r['image']) as im:f=features(enc,pre,[crop_image(im.convert('RGB'),b[:4]) for b in boxes])
  with torch.no_grad():prob=head(f).softmax(1).tolist()
  candidates=[]
  for b,s in zip(boxes,prob):
   c=int(np.argmax(s));candidates.append({'class':c,'xyxy':b[:4],'confidence':b[4]*s[c],'detector_confidence':b[4],'classifier_confidence':s[c]})
  pred=filter_candidates(candidates,point['detector_conf'],point['classifier_conf']);gt=[{'class':o['class_id'],'xyxy':o['bbox_xyxy']} for o in r['objects']];m=match(gt,pred)
  records.append({'id':r['id'],'image_sha256':r['image_sha256'],'label_sha256':r['label_sha256'],'gt':gt,'candidates':candidates,'predictions':pred,'matching':m})
  im=cv2.imread(str(data/r['image']))
  for b in pred:
   xy=[round(v) for v in b['xyxy']];cv2.rectangle(im,tuple(xy[:2]),tuple(xy[2:]),(0,220,255),2);cv2.putText(im,f"{b['class']} {b['confidence']:.2f}",(max(0,xy[0]),max(20,xy[1])),0,.6,(0,0,255),2)
  cv2.imwrite(str(out/(r['id']+'.jpg')),im)
  if idx%20==0:print('test',idx,flush=True)
 stats=eval_detection(records,point['detector_conf'],point['classifier_conf'])
 report={'images':80,'gt_instances_per_class':[80]*4,'stats':stats,'passes':all(s['precision']>.9 and s['recall']>.9 for s in stats),'settings':point,'frozen_inputs':lock,'scope':'fixed before test; no adjustment based on test; same physical objects/setup video holdout, not independent scenes; assistant_visual label review','test_labels_changed':False}
 (out/'predictions.json').write_text(json.dumps(records,indent=2));(out/'report.json').write_text(json.dumps(report,indent=2))
 assert sha(run/'classifier.pt')==lock['classifier_sha256'] and sha(pointpath)==lock['operating_point_sha256']
 print(json.dumps({k:v for k,v in report.items() if k not in ('settings','frozen_inputs')},indent=2),flush=True)
if __name__=='__main__':main()
