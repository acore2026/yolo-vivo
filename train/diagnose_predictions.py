"""Read-only weight/prompt comparison using validation data, not held-out test."""
import os
os.environ['YOLO_CONFIG_DIR']=os.path.abspath('train/.cache/ultralytics')
import json,torch,cv2
from common import ROOT,config
from model_io import load_model
from ultralytics import YOLOWorld
from copy import deepcopy

def main():
 torch.set_num_threads(4);c=config('train/configs/flowers.yaml');w=ROOT/'train/runs/flowers_20261103_191418_402247_full/weights'
 models={'adapter':load_model(ROOT/c['model'],w/'adapter_best.pt'),'merged':YOLOWorld(str(w/'best.pt'))}
 for m in models.values():m.set_classes(c['prompts']);m.model.float().eval()
 torch.manual_seed(42);x=torch.randn(1,3,320,320)
 with torch.no_grad():
  outputs={k:m.model(x)[0] for k,m in models.items()}
 print('RAW',[(k,list(v.shape)) for k,v in outputs.items()],flush=True)
 delta=(outputs['adapter']-outputs['merged']).abs();print('raw max',delta.max().item(),flush=True)
 # Non-tensor runtime attributes affect inference even with matching state_dict.
 attrs=[];bm=dict(models['merged'].model.named_modules())
 for n,m in models['adapter'].model.named_modules():
  if n not in bm:continue
  for k,v in vars(m).items():
   z=getattr(bm[n],k,None)
   if isinstance(v,(int,float,str,bool)) and isinstance(z,(int,float,str,bool)) and v!=z:attrs.append([n,k,v,z])
 root=ROOT/'train/datasets/flowers_video_holdout';rows=json.loads((root/'manifest.json').read_text());rows=[r for r in rows if r['split']=='val']
 results={}
 for tag,m in models.items():
  result=m.val(data=str(root/'data.yaml'),split='val',device='cpu',imgsz=640,batch=8,workers=0,conf=.001,iou=.7,plots=False,project=str(ROOT/'train/diagnostics'),name=f'{tag}_val')
  results[tag]={'metrics':{k:float(v) for k,v in result.results_dict.items()},'per_class':result.summary()}
 report={'raw_max_abs':delta.max().item(),'raw_mean_abs':delta.mean().item(),'attribute_differences':attrs,'validation':results}
 (ROOT/'train/diagnostics/predictions_comparison.json').write_text(json.dumps(report,indent=2,default=lambda v:v.item()))
 print(json.dumps(report,indent=2,default=lambda v:v.item()))
if __name__=='__main__':main()
