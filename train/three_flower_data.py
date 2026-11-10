"""Versioned three-class conversion, preserving original four-class provenance."""
import copy,json,shutil
from collections import Counter
from pathlib import Path
import yaml
from common import ROOT,sha,validate_training_labels
PROMPTS=['potted peony','potted lily','potted chrysanthemum']
NAMES=['芍药','百合','菊花']
TARGET=ROOT/'train/datasets/flowers_three_v1'
SOURCE=ROOT/'train/datasets/flowers_reviewed_standard_v3'

def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def convert(source,target):
 rows=json.loads((source/'manifest.json').read_text());target.mkdir(exist_ok=False,parents=True);kept=[];excluded=[]
 for old in rows:
  assert old['status']=='accepted' and old['review']['reviewer']=='assistant_visual'
  for k in ['image','label']:assert sha(source/old[k])==old[k+'_sha256']
  indices=[i for i,o in enumerate(old['objects']) if o['class_id']<3]
  if old['objects'] and not indices:excluded.append(old['id']);continue
  r=copy.deepcopy(old);r['conversion']={'source_dataset':str(source),'source_manifest_sha256':sha(source/'manifest.json'),'source_label_sha256':old['label_sha256'],'source_review_reused':True,'new_visual_review':False,'removed_class_id':3}
  r['objects']=[dict(old['objects'][i],class_name=NAMES[old['objects'][i]['class_id']],prompt=PROMPTS[old['objects'][i]['class_id']]) for i in indices]
  for k in ['image','label']:(target/r[k]).parent.mkdir(parents=True,exist_ok=True)
  shutil.copy2(source/r['image'],target/r['image']);lines=(source/old['label']).read_text().splitlines();(target/r['label']).write_text(''.join(lines[i]+'\n' for i in indices));r['label_sha256']=sha(target/r['label']);kept.append(r)
 dump(target/'manifest.json',kept);dump(target/'conversion.json',{'source':str(source),'source_manifest_sha256':sha(source/'manifest.json'),'excluded_eucalyptus_only_ids':excluded,'counts':dict(Counter(r['split'] for r in kept)),'review':'Existing accepted reviews reused for deterministic class removal; not newly visually accepted'})
 splits=set(r['split'] for r in kept);(target/'data.yaml').write_text(yaml.safe_dump(dict(path=str(target),**{s:'images/'+s for s in sorted(splits)},names=PROMPTS,partition_policy='same_setup_video_holdout'),sort_keys=False))
 verify(target);return kept

def verify(target):
 target=Path(target);meta=json.loads((target/'conversion.json').read_text());source=Path(meta['source']);assert sha(source/'manifest.json')==meta['source_manifest_sha256'];old={r['id']:r for r in json.loads((source/'manifest.json').read_text())};rows=json.loads((target/'manifest.json').read_text());videos={};hashes={}
 expected={k for k,r in old.items() if not r['objects'] or any(o['class_id']<3 for o in r['objects'])};assert {r['id'] for r in rows}==expected and len(rows)==len(expected)
 for r in rows:
  o=old[r['id']];assert r['status']=='accepted' and r['review']==o['review'];assert r['split']==o['split'] and r['source_sha256']==o['source_sha256'] and r['frame']==o['frame']
  assert sha(target/r['image'])==sha(source/o['image'])==r['image_sha256'];assert sha(target/r['label'])==r['label_sha256']
  objects=[b for b in o['objects'] if b['class_id']<3];assert len(objects)==len(r['objects'])
  for a,b in zip(objects,r['objects']):assert a['class_id']==b['class_id'] and a['bbox_xyxy']==b['bbox_xyxy'] and b['prompt']==PROMPTS[b['class_id']]
  if not objects:assert not o['objects'] and o.get('negative_verified') is True and o['review']['no_target_visible'] is True and r['split']=='train'
  videos.setdefault(r['source_sha256'],set()).add(r['split']);hashes.setdefault(r['image_sha256'],set()).add(r['split'])
 assert all(len(s)==1 for s in list(videos.values())+list(hashes.values()));validate_training_labels(rows,target)
 report={'images':dict(Counter(r['split'] for r in rows)),'objects':{s:dict(Counter(b['class_id'] for r in rows if r['split']==s for b in r['objects'])) for s in {r['split'] for r in rows}},'negative_train':sum(not r['objects'] for r in rows),'source_labels_unchanged':True,'retained_coordinates_unchanged':True,'removed_class_absent':True,'video_split_isolated':True,'independent_scene_split':False}
 return rows,report

if __name__=='__main__':
 rows=convert(SOURCE,TARGET);convert(ROOT/'train/datasets/flowers_test_reviewed_v3',ROOT/'train/datasets/flowers_three_test_v1')
 _,report=verify(TARGET);dump(TARGET/'validation_report.json',report)
 c=yaml.safe_load((ROOT/'train/workflows/20261104_100817_929606/full.yaml').read_text());c.update(data=str((TARGET/'data.yaml').relative_to(ROOT)),prompts=PROMPTS,display_names=NAMES,name='flowers_three_v1',three_class_conversion=True)
 (ROOT/'train/configs/flowers_three_v1.yaml').write_text(yaml.safe_dump(c,allow_unicode=True,sort_keys=False));print(json.dumps(report,ensure_ascii=False,indent=2))
