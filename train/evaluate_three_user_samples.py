"""Paired three-class regression using frozen reviewed GT, no threshold tuning."""
import argparse,json
from pathlib import Path
from common import ROOT,sha,resolve
from audit_validation import match
from three_flower_data import dump,NAMES

def evaluate(gtrows,preds):
 stats=[dict(class_id=c,tp=0,fp=0,fn=0) for c in range(3)];events=[]
 for r in gtrows:
  gt=[b for b in r['gt'] if b['class']<3];pred=[b for b in preds[0 if r['frame'] is None else r['frame']]['detections'] if b['class']<3];result=match(gt,pred)
  for e in result['predictions']:stats[pred[e['prediction']]['class']]['tp' if e['kind']=='tp' else 'fp']+=1
  for j in result['missed_gt']:stats[gt[j]['class']]['fn']+=1
  events.append(dict(frame=r['frame'],matching=result,gt=gt,predictions=pred))
 for s in stats:s.update(precision=s['tp']/max(1,s['tp']+s['fp']),recall=s['tp']/max(1,s['tp']+s['fn']))
 return dict(stats=stats,passes=all(s[k]>.9 for s in stats for k in ['precision','recall']),events=events)
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();out=resolve(a.run);old=ROOT/'train/diagnostics/flower_user_test_20261105_v3';old2=ROOT/'train/diagnostics/flower_test2_20261106_retained_v3'
 annotations=json.loads((old/'review/annotations.json').read_text());test2=ROOT/'train/datasets/flowers_three_test2_samples/annotations.json';ann2=json.loads(test2.read_text());report={'scope':'Known exposed regression, IoU>=0.5; photo plus 12 frames per video, not full-video accuracy','models':{},'gt_hashes':{'original':sha(old/'review/annotations.json'),'test2':sha(test2)}}
 for r in annotations+ann2:assert r['review']=='assistant_visual' and sha(ROOT/r['source'])==r['source_sha256']
 for r in annotations:assert sha(old/'review'/r['preview'])==r['preview_sha256']
 for r in ann2:assert sha(test2.parent/r['image'])==r['image_sha256']
 for tag,files in [('retained_four_class',[old/'01_test.jsonl',old/'02_test.jsonl',old2/'01_test2.jsonl']),('new_three_class',[out/'01_test.jsonl',out/'02_test.jsonl',out/'03_test2.jsonl'])]:
  report['models'][tag]={}
  for source,gt,file in zip(['photo','test_video_sample','test2_video_sample'],[[r for r in annotations if r['frame'] is None],[r for r in annotations if r['frame'] is not None],ann2],files):
   pred=[json.loads(s) for s in file.read_text().splitlines()];assert all(r['frame']==i for i,r in enumerate(pred));report['models'][tag][source]=evaluate(gt,pred)
 dump(out/'paired_sample_metrics.json',report);lines=['# 三类前后配对回归','',report['scope'],'','旧四类模型只评估前三类输出；任何桉树叶误认成前三类仍计FP。各模型使用自身验证集冻结阈值。','']
 for tag,sources in report['models'].items():
  for source,result in sources.items():
   lines += [f'## {tag} / {source}','','|类别|TP|FP|FN|P|R|','|---|---:|---:|---:|---:|---:|']
   for s in result['stats']:lines.append(f'|{NAMES[s["class_id"]]}|{s["tp"]}|{s["fp"]}|{s["fn"]}|{s["precision"]:.2%}|{s["recall"]:.2%}|')
   lines+=['']
 (out/'COMPARISON_CN.md').write_text('\n'.join(lines))
if __name__=='__main__':main()
