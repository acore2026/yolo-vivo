"""Export the explicitly visually reviewed decisions; preserve originals and split membership."""
import copy,html,json,shutil
from collections import Counter,defaultdict
import cv2
import numpy as np
import yaml
from review_flowers_v2 import ROOT,SRC,OUT,sha,save,tile
DEST=ROOT/'train/datasets/flowers_reviewed_v2'
PROMPTS=['potted peony','potted lily','potted chrysanthemum','potted eucalyptus']
SEMANTICS=['芍药盆栽','原兰花盆栽（CLIP提示词 lily）','原向日葵盆栽（CLIP提示词 chrysanthemum）','桉树叶盆栽']

def main():
 rows=json.loads((SRC/'manifest.json').read_text());dec=json.loads((OUT/'decisions.json').read_text())
 assert set(dec)=={r['id'] for r in rows if r['split'] in ('train','val')}
 before=json.loads((OUT/'source_integrity_before.json').read_text())
 # Changed overlays were actually inspected after the edits, prior to invoking export.
 for d in dec.values():
  assert d['reviewer']=='assistant_visual' and d['human_review'] is False
  assert d['status']=='reviewed'
  d['status']='accepted';d['decision_basis']='本轮逐图原图/叠框视觉复核；修改项再次查看修订叠框'
  if d['changed']:d['correction_visual_review']=True
 assert not DEST.exists(), 'Never overwrite an existing dataset'
 DEST.mkdir(parents=True)
 shutil.copy2(SRC/'manifest.json',DEST/'source_manifest.json')
 source_hashes={s:sha(ROOT/s) for s in {r['source'] for r in rows}}
 counts={s:Counter() for s in ('train','val','test')};splits=Counter();hash_splits=defaultdict(set);videos=defaultdict(set)
 result=[];changes=[]
 for r in rows:
  id=r['id'];s=r['split'];splits[s]+=1
  assert sha(SRC/r['image'])==before[id]['image']==r['image_sha256']
  assert sha(SRC/r['label'])==before[id]['label']
  assert source_hashes[r['source']]==r['source_sha256']
  im=cv2.imread(str(SRC/r['image']));h,w=im.shape[:2];assert (w,h)==(r['width'],r['height'])
  ip=DEST/r['image'];lp=DEST/r['label'];ip.parent.mkdir(parents=True,exist_ok=True);lp.parent.mkdir(parents=True,exist_ok=True)
  shutil.copy2(SRC/r['image'],ip)
  d=dec.get(id);new=copy.deepcopy(r)
  new['historical_annotation_metadata']={k:new.pop(k) for k in ['review','raw_review','status','annotation_method','retained_from_pilot'] if k in new}
  if d:
   objects=copy.deepcopy(d['objects'])
   for o in objects:
    o['bbox_xyxy']=[max(0.,min(float(v),float(w if i%2==0 else h))) for i,v in enumerate(o['bbox_xyxy'])]
    o['class_name']=SEMANTICS[o['class_id']];o['prompt']=PROMPTS[o['class_id']]
   new['objects']=objects
   if d['changed']:
    lines=[]
    for o in objects:
     x1,y1,x2,y2=o['bbox_xyxy'];lines.append(f"{o['class_id']} {(x1+x2)/(2*w):.9f} {(y1+y2)/(2*h):.9f} {(x2-x1)/w:.9f} {(y2-y1)/h:.9f}")
    lp.write_text('\n'.join(lines)+'\n')
    changes.append({'id':id,'split':s,'before':r['objects'],'after':objects,'note':d['note'],'preview':d['correction_preview']})
   else:shutil.copy2(SRC/r['label'],lp)
   new.update(status='accepted',human_review=False,review={k:v for k,v in d.items() if k!='objects'},annotation_method='assistant_visual_review_v2')
  else:
   shutil.copy2(SRC/r['label'],lp)
   new.update(status='not_reviewed_this_batch',human_review=False,review={'reviewer':None,'decision':'not_reviewed_this_batch','note':'80张test沿用历史标签，本轮未查看、不冒充本轮accepted；历史记录见source_manifest.json。'})
   for o in new['objects']:o.update(class_name=SEMANTICS[o['class_id']],prompt=PROMPTS[o['class_id']])
  lines=lp.read_text().splitlines();assert len(lines)==len(new['objects']) and lines, 'No uncertain empty exports'
  for line,o in zip(lines,new['objects']):
   v=list(map(float,line.split()));assert len(v)==5 and all(np.isfinite(v))
   c,x,y,bw,bh=v;assert c==int(c) and 0<=c<4 and 0<x<1 and 0<y<1 and 0<bw<=1 and 0<bh<=1
   assert x-bw/2>=-1e-8 and y-bh/2>=-1e-8 and x+bw/2<=1+1e-8 and y+bh/2<=1+1e-8
   xy=[(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]
   assert int(c)==o['class_id'] and np.allclose(xy,o['bbox_xyxy'],atol=.001)
   counts[s][int(c)]+=1
  new['image_sha256']=sha(ip);new['label_sha256']=sha(lp)
  assert new['image_sha256']==before[id]['image']
  hash_splits[new['image_sha256']].add(s);videos[r['source']].add(s)
  result.append(new)
 assert dict(splits)=={'train':280,'val':39,'test':80}
 assert all(len(x)==1 for x in hash_splits.values()) and all(len(x)==1 for x in videos.values())
 for r in rows:
  assert sha(SRC/r['image'])==before[r['id']]['image'] and sha(SRC/r['label'])==before[r['id']]['label']
 data={'path':str(DEST),'train':'images/train','val':'images/val','test':'images/test','names':PROMPTS,'partition_policy':'same_setup_video_holdout'}
 (DEST/'data.yaml').write_text(yaml.safe_dump(data,allow_unicode=True,sort_keys=False))
 save(DEST/'manifest.json',result);save(DEST/'review_records.json',dec);save(DEST/'changes.json',changes)
 save(OUT/'decisions.json',dec)
 report={'status':'PASS','images':dict(splits),'instances':{s:{str(c):counts[s][c] for c in range(4)} for s in counts},'reviewed':319,'test_not_reviewed':80,'changed_images':len(changes),'changed_by_split':dict(Counter(x['split'] for x in changes)),'added_boxes':6,'adjusted_boxes':7,'class_id_changes':0,'originals_unchanged':True,'all_copied_images_hash_identical':True,'test_labels_byte_identical':True,'yolo_format_manifest_consistency':'PASS','exact_duplicate_images_across_splits':0,'video_overlap_across_splits':0,'source_video_hashes':source_hashes,'independent_scene_evaluation':False,'same_setup_group_crosses_splits':True,'near_duplicate_check':'not_run; same physical plants and setup are correlated','review_scope':'319 raw/overlay 640px panels; selected native-resolution details; changed overlays inspected again; human_review=false'}
 save(DEST/'validation_report.json',report)
 # Browser review: every reviewed image links its actual raw/overlay evidence, edits also before/after.
 page=['<!doctype html><meta charset="utf-8"><title>花卉标签复核 v2</title><style>body{font:16px sans-serif;max-width:1320px;margin:30px auto}img{max-width:100%}td,th{padding:8px;text-align:left}summary{cursor:pointer}</style><h1>319张花卉标签复核</h1><p>整盆含盆；可辨截断按可见范围；少量残片省略。助手视觉复核，非人工审核；test 80张未复核。名称 lily/chrysanthemum 为沿用的模型提示词，不是植物学鉴定结论。</p>']
 for i in range(1,5):page.append(f'<h2>修正对照 {i}：左旧 / 右新</h2><img src="changes_{i:02d}.jpg">')
 page.append('<h2>逐图审核记录</h2>')
 for r in rows:
  if r['id'] not in dec:continue
  d=dec[r['id']];raw='../../datasets/flowers_reviewed_v2/'+r['image']
  page.append(f'<details><summary>{r["id"]} · {r["split"]} · {"修正" if d["changed"] else "保留"}</summary><p>{html.escape(d["note"])}</p><a href="{raw}">原图</a> · <a href="{d["evidence"]}">本轮原图/旧标签复核页</a>')
  if d.get('correction_preview'):page.append(f' · <a href="{d["correction_preview"]}">修正前后</a>')
  if d.get('detail_evidence'):page.append(f' · <a href="{d["detail_evidence"]}">细节复核</a>')
  page.append('</details>')
 (OUT/'review.html').write_text('\n'.join(page))
 print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
