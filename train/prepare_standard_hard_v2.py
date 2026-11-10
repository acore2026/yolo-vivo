"""Export manually reviewed supplementary train frames; preserve previous annotations.
Coordinates below were independently drawn from the 640x360 raw review tiles.
Run only after the accompanying raw/overlay review is complete.
"""
import json,shutil,sys
from pathlib import Path
from collections import Counter
from PIL import Image,ImageDraw
import yaml
from common import ROOT,sha,validate_training_labels
BATCH=ROOT/'train/diagnostics/flower_hard_batch_v2'
# class, visible whole-pot bounds in 640x360 local review coordinates.
BOXES=[
[(2,0,98,241,307),(0,164,70,307,197),(1,305,20,418,145),(3,422,52,548,180)],
[(2,30,43,251,241),(0,163,44,289,159),(1,285,15,397,123),(3,392,56,496,147)],
[(2,14,78,252,284),(0,188,43,333,174),(1,329,10,448,135),(3,438,53,586,196)],
[(0,92,45,322,248),(1,330,0,501,179),(3,501,63,640,284)],
[(0,0,143,185,360),(1,194,58,419,281),(3,422,177,640,360)],
[(1,89,80,273,273),(3,276,113,473,322)],
[(0,93,50,286,216),(1,300,12,439,164),(3,407,58,610,239)],
[(2,0,42,299,304),(0,315,25,530,213),(1,508,4,640,196)],
[(2,52,9,356,285),(0,313,26,472,165),(1,444,0,595,130)],
[(2,0,89,226,360),(0,165,71,339,224),(1,334,33,478,198),(3,474,89,640,226)],
[(2,22,106,195,249),(0,158,94,278,186),(1,261,69,340,148),(3,329,86,418,157)],
[(2,4,92,151,214),(0,148,76,246,179),(1,240,70,317,151),(3,304,68,381,160)],
[(0,49,115,327,360)],[(0,147,149,312,357)],
[(1,96,48,358,324)],[(1,118,110,343,333)],
[(2,96,20,430,285)],[(2,0,13,338,272)],
[(3,207,96,386,277)],[(3,87,90,255,280)]
]
def main():
 rows=json.loads((BATCH/'pending_manifest.json').read_text()); assert len(rows)==len(BOXES)
 for i,(r,boxes) in enumerate(zip(rows,BOXES)):
  im=Image.open(BATCH/'images'/f'{r["id"]}.png').convert('RGB'); im.thumbnail((640,360));d=ImageDraw.Draw(im)
  for cls,x1,y1,x2,y2 in boxes:
   d.rectangle((x1,y1,x2,y2),outline=['red','magenta','yellow','cyan'][cls],width=2); d.text((x1+2,y1+2),str(cls),fill='black',stroke_width=1,stroke_fill='white')
  im.save(BATCH/f'overlay_{i+1:02d}.jpg',quality=95)
 for k in range(0,20,4):
  sheet=Image.new('RGB',(1280,776),'#202020'); d=ImageDraw.Draw(sheet)
  for j,r in enumerate(rows[k:k+4]):
   x=j%2*640;y=j//2*388;sheet.paste(Image.open(BATCH/f'overlay_{k+j+1:02d}.jpg'),(x,y+28));d.text((x+5,y+5),r['id'],fill='white')
  sheet.save(BATCH/f'boxes_{k//4+1:02d}.jpg',quality=95)
 if '--export' not in sys.argv:return
 target=ROOT/'train/datasets/flowers_reviewed_standard_v2';target.mkdir(exist_ok=False)
 old=ROOT/'train/datasets/flowers_reviewed_v2_trainval'; existing=json.loads((old/'manifest.json').read_text())
 for r in existing:
  for key in ['image','label']:
   dest=target/r[key];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old/r[key],dest)
 for i,(r,boxes) in enumerate(zip(rows,BOXES)):
  r.update(image=f'images/train/{r["id"]}.png',label=f'labels/train/{r["id"]}.txt',human_review=False,objects=[dict(class_id=b[0],bbox_xyxy=[v*2 for v in b[1:]]) for b in boxes],status='accepted',annotation_method='manual_visible_whole_pot_xyxy_640px_review_scale2')
  r['review']=dict(reviewer='assistant_visual',human_review=False,status='accepted',raw_evidence=f'raw_{i//4+1:02d}.jpg',overlay_evidence=f'boxes_{i//4+1:02d}.jpg',tile_index=i%4,evidence_directory=str(BATCH.relative_to(ROOT)),scope='All raw and overlay 640x360 tiles viewed; manually drawn boxes, no inherited candidate coordinates',note='可辨多数可见的截断目标按可见范围标注；极小边缘局部不标。四类按用户原命名。')
  dest=target/r['image'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BATCH/'images'/f'{r["id"]}.png',dest)
  label=target/r['label'];label.parent.mkdir(parents=True,exist_ok=True)
  lines=[]
  for cls,x1,y1,x2,y2 in boxes:lines.append(f'{cls} {(x1+x2)/1280:.9f} {(y1+y2)/720:.9f} {(x2-x1)/640:.9f} {(y2-y1)/360:.9f}')
  label.write_text('\n'.join(lines)+'\n');r['label_sha256']=sha(label)
 combined=existing+rows;validate_training_labels(combined,target)
 assert len({(r['source_sha256'],r['frame']) for r in combined})==len(combined)
 assert len({r['image_sha256'] for r in combined})==len(combined)
 (target/'manifest.json').write_text(json.dumps(combined,ensure_ascii=False,indent=2)+'\n')
 names=['peony_pot','orchid_pot','sunflower_pot','eucalyptus_pot']
 (target/'data.yaml').write_text(yaml.safe_dump(dict(path=str(target),train='images/train',val='images/val',names=names),sort_keys=False))
 report=dict(images=dict(Counter(r['split'] for r in combined)),new_images=len(rows),new_objects=dict(Counter(b[0] for bb in BOXES for b in bb)),label_format='PASS',unique_source_frames='PASS',unique_image_hashes='PASS',unchanged_val_images=39,source_manifest_sha256=sha(old/'manifest.json'),new_manifest_sha256=sha(target/'manifest.json'),review='assistant_visual, not human',independent_scene_split=False,unresolved=['Same plants/setup and nearby frames: limited new diversity','Small val class counts; 90% cannot be guaranteed on unseen scenes','Botanical names follow user, not independent botanical authentication'])
 (target/'validation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 (BATCH/'reviewed_manifest.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
