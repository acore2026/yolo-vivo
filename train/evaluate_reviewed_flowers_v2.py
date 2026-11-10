"""Compare three frozen adapters on identical v2 validation images/labels; no training."""
import json,os
os.environ.setdefault('MPLCONFIGDIR','/tmp/flowers-v2-matplotlib')
import yaml,torch
from common import ROOT,sha
from model_io import load_model
from audit_validation import match
DATA=ROOT/'train/datasets/flowers_reviewed_v2'
OUT=ROOT/'train/diagnostics/reviewed_v2_evaluation'
ADAPTERS={
 'previous':('train/diagnostics/adapter_best_architecture_v2.pt',['potted peony','potted orchid','potted sunflower','potted eucalyptus']),
 'strongaug':('train/runs/flowers_20261103_200348_146656_full/weights/adapter_best.pt',['potted peony','potted orchid','potted sunflower','potted eucalyptus']),
 'clip_names':('train/runs/flowers_20261103_210003_062186_full/weights/adapter_best.pt',['potted peony','potted lily','potted chrysanthemum','potted eucalyptus'])}

def main():
 OUT.mkdir(exist_ok=False);torch.set_num_threads(4)
 rows=[r for r in json.loads((DATA/'manifest.json').read_text()) if r['split']=='val']
 report={'data_yaml':str(DATA/'data.yaml'),'manifest_sha256':sha(DATA/'manifest.json'),'split':'val','image_count':len(rows),'settings':{'imgsz':640,'conf_AP':.001,'fixed_conf':.25,'nms_iou':.7,'match_iou':.5,'device':'cpu','seed':42},'models':{}}
 for tag,(path,names) in ADAPTERS.items():
  cfg=yaml.safe_load((DATA/'data.yaml').read_text());cfg['names']=names
  config=OUT/(tag+'_data.yaml');config.write_text(yaml.safe_dump(cfg,allow_unicode=True))
  model=load_model(ROOT/'models/yolov8s-worldv2.pt',ROOT/path);model.set_classes(names)
  res=model.val(data=str(config),split='val',imgsz=640,batch=8,device='cpu',workers=0,conf=.001,iou=.7,project=str(OUT),name=tag,plots=True,verbose=False)
  metrics={k:float(v) for k,v in res.results_dict.items()}
  ap=[{'class_id':int(c),'prompt':names[int(c)],'precision_bestF1':float(res.box.p[i]),'recall_bestF1':float(res.box.r[i]),'ap50':float(res.box.ap50[i]),'ap50_95':float(res.box.ap[i])} for i,c in enumerate(res.box.ap_class_index)]
  stats=[{'class_id':i,'gt':0,'tp':0,'fp':0,'fn':0} for i in range(4)];predrows=[]
  for row in rows:
   pred=model.predict(str(DATA/row['image']),imgsz=640,device='cpu',conf=.25,iou=.7,verbose=False)[0]
   pred=[{'class':int(b[5]),'confidence':float(b[4]),'xyxy':b[:4]} for b in pred.boxes.data.cpu().tolist()]
   gt=[{'class':o['class_id'],'xyxy':o['bbox_xyxy']} for o in row['objects']];matched=match(gt,pred)
   for g in gt:stats[g['class']]['gt']+=1
   for e in matched['predictions']:stats[pred[e['prediction']]['class']]['tp' if e['kind']=='tp' else 'fp']+=1
   for j in matched['missed_gt']:stats[gt[j]['class']]['fn']+=1
   predrows.append({'id':row['id'],'image_sha256':row['image_sha256'],'label_sha256':row['label_sha256'],'gt':gt,'predictions':pred,'matching':matched})
  for s in stats:
   s['precision']=s['tp']/max(1,s['tp']+s['fp']);s['recall']=s['tp']/max(1,s['gt'])
  (OUT/(tag+'_predictions.json')).write_text(json.dumps(predrows,indent=2))
  report['models'][tag]={'adapter':path,'adapter_sha256':sha(ROOT/path),'prompts':names,'metrics':metrics,'per_class':ap,'fixed_conf_025':stats}
  (OUT/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
  print(tag,json.dumps(metrics),flush=True);del model
 print('COMPLETED',OUT,flush=True)
if __name__=='__main__':main()
