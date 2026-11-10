import json
from pathlib import Path
from PIL import Image,ImageDraw
root=Path('/home/huawei/VIVO/yolo-vivo');data=root/'train/datasets/flowers_reviewed_standard_v2';out=root/'train/diagnostics/standard_yolo_v3_audit'
rows=[r for r in json.loads((data/'manifest.json').read_text()) if r['split']=='train' and any(b['class_id']==3 for b in r['objects'])];rows.sort(key=lambda r:(r['source'],r['frame']))
for start in range(0,len(rows),16):
 sheet=Image.new('RGB',(1280,208*8),'#202020');d=ImageDraw.Draw(sheet)
 for j,r in enumerate(rows[start:start+16]):
  x=j%2*640;y=j//2*208;im=Image.open(data/r['image']).convert('RGB').resize((320,180));box=im.copy();bd=ImageDraw.Draw(box)
  for b in r['objects']:
   if b['class_id']==3:bd.rectangle([v/4 for v in b['bbox_xyxy']],outline='cyan',width=2)
  sheet.paste(im,(x,y+28));sheet.paste(box,(x+320,y+28));d.text((x+3,y+6),f'{start+j:03d} {r["id"]}',fill='white')
 sheet.save(out/f'audit_{start//16+1:02d}.jpg',quality=96)
(out/'audit_index.json').write_text(json.dumps([dict(id=r['id'],image=r['image'],image_sha256=r['image_sha256'],page=i//16+1,tile=i%16) for i,r in enumerate(rows)],indent=2))
print('125 eucalyptus positive frames, raw/box paired sheets')
