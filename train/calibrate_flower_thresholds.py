"""Per-class validation calibration; always reevaluate joint NMS/matching."""
import json,argparse
from pathlib import Path
from flower_classifier import eval_detection,macrof1
from common import sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=Path(a.run);out=run/'per_class_calibration';out.mkdir(exist_ok=False)
 rows=json.loads((run/'val_candidates.json').read_text());grid=[]
 for d in [.01,.025,.05,.1,.15,.25]:
  for c in [0,.3,.35,.4,.45,.5,.6,.7,.85,.95]:
   stats=eval_detection(rows,d,c);grid.append({'detector_conf':d,'classifier_conf':c,'stats':stats,'macro_f1':macrof1(stats)})
 points=[]
 for cls in range(4):
  def key(g):
   s=g['stats'][cls];return (min(s['precision'],s['recall']),2*s['tp']/max(1,2*s['tp']+s['fp']+s['fn']),g['macro_f1'])
  points.append(max(grid,key=key))
 ds=[x['detector_conf'] for x in points];cs=[x['classifier_conf'] for x in points];stats=eval_detection(rows,ds,cs)
 point={'detector_conf':ds,'classifier_conf':cs,'nms_iou':.5,'stats':stats,'macro_f1':macrof1(stats),'min_precision_recall':min(min(s['precision'],s['recall']) for s in stats),'passes':all(s['precision']>.9 and s['recall']>.9 for s in stats),'scope':'selected on val, joint NMS evaluated; not final test','classifier_sha256':sha(run/'classifier.pt'),'candidate_sha256':sha(run/'val_candidates.json')}
 (out/'threshold_grid.json').write_text(json.dumps(grid,indent=2));(out/'operating_point.json').write_text(json.dumps(point,indent=2));print(json.dumps(point,indent=2))
if __name__=='__main__':main()
