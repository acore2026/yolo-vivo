"""Frozen operating-point regression on an existing reviewed set; no tuning."""
import argparse,json,os
from pathlib import Path
from common import ROOT,sha,resolve,validate_training_labels

def main():
 p=argparse.ArgumentParser();p.add_argument('--workflow',required=True);p.add_argument('--dataset',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 workflow=resolve(a.workflow);dataset=resolve(a.dataset);out=resolve(a.output)
 state=json.loads((workflow/'status.json').read_text());point=json.loads((workflow/'operating_point.json').read_text());ck=resolve(state['best_checkpoint']);assert state['status']=='completed';assert sha(ck)==point['checkpoint_sha256']==state['best_checkpoint_sha256']
 rows=json.loads((dataset/'manifest.json').read_text());validate_training_labels(rows,dataset)
 for r in rows:
  assert r['status']=='accepted' and r['review']['reviewer']=='assistant_visual'
  for k in ['image','label']:assert sha(dataset/r[k])==r[k+'_sha256']
 train=json.loads((resolve(state['configuration']['source_dataset'])/'manifest.json').read_text());assert not {r['source_sha256'] for r in rows}&{r['source_sha256'] for r in train}
 out.mkdir(parents=True,exist_ok=False);os.environ['YOLO_CONFIG_DIR']=str(ROOT/'train/.cache/standard_yolo');os.environ['YOLO_AUTOINSTALL']='false'
 import torch
 torch.set_num_threads(4)
 from ultralytics import YOLO
 from audit_validation import match
 from evaluate_user_flower_test import metrics
 model=YOLO(str(ck));records=[]
 for r in rows:
  result=model.predict(str(dataset/r['image']),device='cpu',imgsz=point['imgsz'],conf=point['conf'],iou=point['nms_iou'],agnostic_nms=point['agnostic_nms'],max_det=100,verbose=False)[0]
  pred=[dict(class_id=int(b[5]),xyxy=b[:4],confidence=b[4]) for b in result.boxes.data.cpu().tolist()];pred=[{'class':b['class_id'],'xyxy':b['xyxy'],'confidence':b['confidence']} for b in pred]
  gt=[{'class':b['class_id'],'xyxy':b['bbox_xyxy']} for b in r['objects']];records.append(dict(id=r['id'],gt=gt,predictions=pred,matching=match(gt,pred)))
 stats=metrics(records);report=dict(scope='Existing previously exposed flower3 regression, not new unseen scene; reuse prior reviewed labels, no new visual acceptance',dataset=str(dataset),manifest_sha256=sha(dataset/'manifest.json'),model_sha256=sha(ck),operating_point=point,images=len(rows),stats=stats,passes=all(s[k] is not None and s[k]>.9 for s in stats for k in ['precision','recall']),records=records)
 (out/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))
if __name__=='__main__':main()
