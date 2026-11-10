"""Compare frozen tensors, adapter reconstruction and trained merged weights on CPU."""
from pathlib import Path
import json
import torch
from common import ROOT
from model_io import load_model
from ultralytics import YOLOWorld

def main():
 torch.set_num_threads(4)
 w=ROOT/'train/runs/flowers_20261103_191418_402247_full/weights'
 base=load_model(ROOT/'models/yolov8s-worldv2.pt').model.float().eval()
 adapter=load_model(ROOT/'models/yolov8s-worldv2.pt',w/'adapter_best.pt').model.float().eval()
 merged=YOLOWorld(str(w/'best.pt')).model.float().eval()
 a=adapter.state_dict();m=merged.state_dict();b=base.state_dict()
 diffs=[]
 for k in a.keys()&m.keys():
  if a[k].shape!=m[k].shape:continue
  delta=(a[k].float()-m[k].float()).abs()
  if delta.max()>0:diffs.append({'name':k,'max_abs':delta.max().item(),'mean_abs':delta.mean().item(),'shape':list(a[k].shape)})
 state=torch.load(w/'training_state_last.pt',map_location='cpu',weights_only=False)
 frozen=[]
 for k,v in state['model'].items():
  if k.endswith(('lora_A','lora_B')):continue
  original=k.replace('.base.','.')
  if original in b and b[original].shape==v.shape:
   d=(v.float()-b[original].float()).abs().max().item()
   if d>0:frozen.append({'name':k,'max_abs':d})
 report={'adapter_names':torch.load(w/'adapter_best.pt',weights_only=True)['names'],
 'merged_names':merged.names,'top_adapter_vs_merged_differences':sorted(diffs,key=lambda x:x['max_abs'],reverse=True)[:35],
 'frozen_training_vs_original_differences':frozen}
 p=ROOT/'train/diagnostics/weights_comparison.json';p.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
