"""Prepare pending train-only background crops; export only visually approved IDs.
No source, test, model or previous dataset is overwritten.
"""
import argparse,copy,json,shutil
from pathlib import Path
from PIL import Image,ImageDraw
import yaml
from common import ROOT,sha,validate_training_labels
SOURCE=ROOT/'train/datasets/flowers_reviewed_standard_v2'
BATCH=ROOT/'train/diagnostics/standard_yolo_v3_audit'
TARGET=ROOT/'train/datasets/flowers_reviewed_standard_v3'
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def prepare():
 rows=json.loads((SOURCE/'manifest.json').read_text());out=BATCH/'negative_candidates';out.mkdir(exist_ok=True);new=[]
 for video in [1,4,5,6,7]:
  pool=sorted([r for r in rows if r['split']=='train' and r['source']==f'data/flower{video}.mp4'],key=lambda r:r['frame'])
  assert sha(ROOT/pool[0]['source'])==pool[0]['source_sha256']
  for j in range(4):
   center=round((j+.5)*len(pool)/4)-1;rects=[]
   for index in sorted(range(j*len(pool)//4,(j+1)*len(pool)//4),key=lambda k:abs(k-center)):
    r=pool[index]
    for width in [640,320]:
     for y in [480,240,0]:
      for x in range(0,1281-width,320):
       rect=[x,y,x+width,y+240]
       if all(rect[2]<b['bbox_xyxy'][0]-20 or rect[0]>b['bbox_xyxy'][2]+20 or rect[3]<b['bbox_xyxy'][1]-20 or rect[1]>b['bbox_xyxy'][3]+20 for b in r['objects']):rects.append(rect)
     if rects:break
    if rects:break
   assert rects,r['id'];rect=rects[j%min(3,len(rects))];im=Image.open(SOURCE/r['image']);assert sha(SOURCE/r['image'])==r['image_sha256']
   ident=f'{r["id"]}_bg_{rect[0]}_{rect[1]}';dest=out/f'{ident}.png';crop=im.crop(rect)
   if dest.exists():assert Image.open(dest).tobytes()==crop.tobytes()
   else:crop.save(dest)
   new.append(dict(id=ident,parent_id=r['id'],parent_image=r['image'],parent_image_sha256=r['image_sha256'],source=r['source'],source_sha256=r['source_sha256'],frame=r['frame'],source_fps=r['source_fps'],source_total_frames=r['source_total_frames'],timestamp_seconds=r['timestamp_seconds'],group=r['group'],split='train',width=crop.width,height=crop.height,source_width=r['width'],source_height=r['height'],crop_xyxy=rect,image_sha256=sha(dest),status='pending',objects=[]))
 write(BATCH/'negative_pending.json',new)
 for start in range(0,len(new),8):
  sheet=Image.new('RGB',(1280,4*268),'#202020');d=ImageDraw.Draw(sheet)
  for j,r in enumerate(new[start:start+8]):
   x=j%2*640;y=j//2*268;sheet.paste(Image.open(out/f'{r["id"]}.png'),(x,y+28));d.text((x+5,y+6),f'{start+j:02d} {r["id"]}',fill='white')
  sheet.save(BATCH/f'negative_{start//8+1:02d}.jpg',quality=96)
 print('20 pending crops; labels NOT exported')
def export():
 rows=json.loads((SOURCE/'manifest.json').read_text());pending=json.loads((BATCH/'negative_pending.json').read_text());reviews=json.loads((BATCH/'negative_review.json').read_text());byid={r['id']:r for r in pending};accepted=[]
 for review in reviews:
  assert review['reviewer']=='assistant_visual' and review['decision']=='accepted_negative' and review['no_target_visible'] is True
  r=copy.deepcopy(byid[review['id']]);r.update(status='accepted',human_review=False,review=review,negative_verified=True,annotation_type='reviewed_background_crop',derivation={'type':'crop','parent_id':r['parent_id'],'crop_xyxy':r['crop_xyxy']},image=f'images/train/{r["id"]}.png',label=f'labels/train/{r["id"]}.txt');accepted.append(r)
 assert accepted;TARGET.mkdir(exist_ok=False)
 for r in rows:
  for key in ['image','label']:
   dest=TARGET/r[key];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(SOURCE/r[key],dest)
 for r in accepted:
  shutil.copy2(BATCH/'negative_candidates'/f'{r["id"]}.png',TARGET/r['image']);(TARGET/r['label']).write_text('');r['label_sha256']=sha(TARGET/r['label'])
 combined=rows+accepted;write(TARGET/'manifest.json',combined);validate_training_labels(combined,TARGET)
 config=yaml.safe_load((ROOT/'train/configs/flowers_standard_yolo_v2.yaml').read_text());config.update(source_dataset=str(TARGET.relative_to(ROOT)),dataset='train/datasets/flowers_standard_yolo_v3',expected_images={'train':300+len(accepted),'val':39},source_manifest_sha256=sha(TARGET/'manifest.json'),workflow_prefix='standard_yolo_v3',allow_reviewed_background_crops=True)
 (ROOT/'train/configs/flowers_standard_yolo_v3.yaml').write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False))
 (TARGET/'data.yaml').write_text(yaml.safe_dump(dict(path=str(TARGET),train='images/train',val='images/val',names=config['names']),sort_keys=False))
 write(TARGET/'preparation_report.json',dict(positive_train=300,negative_train=len(accepted),val_unchanged=39,negative_scope='Train-only parent-image crops, manually reviewed no visible target; not new independent captures',source_manifest_sha256=sha(SOURCE/'manifest.json'),manifest_sha256=sha(TARGET/'manifest.json'),augmentation='Only train; same online augmentation as v2',remaining_issues=['Same plants and tabletop scenes','No new top-down positive captures or independent background footage','Cannot promise P/R >90%']))
 print('Exported',len(accepted),'reviewed negatives')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--export',action='store_true');a=p.parse_args();export() if a.export else prepare()
