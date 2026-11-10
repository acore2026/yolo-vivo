"""Train-only candidate crops for visual review; never automatically accept."""
import json,math
import numpy as np
from PIL import Image,ImageOps,ImageDraw
from flower_classifier import DATA,ROOT
from common import sha
OUT=ROOT/'train/diagnostics/flower_hard_review_v2'
def main():
 OUT.mkdir(exist_ok=False);(OUT/'crops').mkdir()
 rows=[r for r in json.loads((DATA/'manifest.json').read_text()) if r['split']=='train'];selected=[]
 for c,video in enumerate(['flower4','flower5','flower6','flower7']):
  group=[r for r in rows if r['id'].startswith(video+'_')]
  for i in np.linspace(0,len(group)-1,4,dtype=int):selected.append((c,group[i]))
 records=[];tiles=[]
 for c,r in selected:
  im=Image.open(DATA/r['image']).convert('RGB');o=next(o for o in r['objects'] if o['class_id']==c);x1,y1,x2,y2=o['bbox_xyxy'];w=x2-x1;h=y2-y1
  candidates=[('pot_fragment',[x1,y1+.78*h,x2,y2],4),('background',[im.width-280,0,im.width,200],4),('truncated',[x1+.18*w,y1,x2,y2],c)]
  for kind,box,label in candidates:
   box=[max(0,math.floor(box[0])),max(0,math.floor(box[1])),min(im.width,math.ceil(box[2])),min(im.height,math.ceil(box[3]))]
   crop=im.crop(tuple(box));id=f'hard_{len(records):03d}';path=OUT/'crops'/f'{id}.png';crop.save(path)
   records.append({'id':id,'source_id':r['id'],'source_image':r['image'],'source_image_sha256':r['image_sha256'],'source_video':r['source'],'frame':r['frame'],'split':'train','bbox_xyxy':box,'kind':kind,'proposed_class':label,'status':'pending','crop':str(path.relative_to(OUT)),'crop_sha256':sha(path)})
   tile=Image.new('RGB',(320,270),'white');small=ImageOps.contain(crop,(320,230));tile.paste(small,((320-small.width)//2,0));d=ImageDraw.Draw(tile);d.text((5,235),f'{id} {kind} proposed={label}',fill='black');d.text((5,250),r['id'],fill='black');tiles.append(tile)
 for i in range(0,len(tiles),8):
  page=Image.new('RGB',(1280,540),'white')
  for j,t in enumerate(tiles[i:i+8]):page.paste(t,((j%4)*320,(j//4)*270))
  page.save(OUT/f'page_{i//8+1:02d}.jpg')
 (OUT/'candidates.json').write_text(json.dumps(records,indent=2));print(len(records))
if __name__=='__main__':main()
