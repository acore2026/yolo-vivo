"""Mine real TRAIN detections for visual review. No automatic acceptance."""
import json,math
import torch,numpy as np,cv2
from PIL import Image,ImageOps,ImageDraw
from flower_classifier import *
OUT=ROOT/'train/diagnostics/flower_candidate_review_v3'
def main():
 OUT.mkdir(exist_ok=False);(OUT/'crops').mkdir();torch.set_num_threads(4)
 detector=load_model(ROOT/'models/yolov8s-worldv2.pt',ADAPTER);detector.set_classes(PROMPTS)
 rows=[r for r in json.loads((DATA/'manifest.json').read_text()) if r['split']=='train'];pool=[]
 for idx,r in enumerate(rows):
  pred=detector.predict(str(DATA/r['image']),device='cpu',imgsz=640,conf=.01,iou=.7,verbose=False)[0].boxes.data.cpu().tolist()
  selected=[]
  for b in sorted(pred,key=lambda b:-b[4]):
   if any(iou(b[:4],x[:4])>.65 for x in selected):continue
   selected.append(b)
   overlaps=[iou(b[:4],o['bbox_xyxy']) for o in r['objects']];best=max(overlaps,default=0);g=r['objects'][int(np.argmax(overlaps))] if overlaps else None
   x1,y1,x2,y2=b[:4];edge=min(x1,y1,1280-x2,720-y2)<6
   if not edge and best>.5:continue
   pool.append({'source_id':r['id'],'source_image':r['image'],'source_image_sha256':r['image_sha256'],'source':r['source'],'source_sha256':r['source_sha256'],'frame':r['frame'],'split':'train','bbox_xyxy':b[:4],'detector_class':int(b[5]),'detector_confidence':b[4],'edge':edge,'best_gt_iou':best,'proposed_class':g['class_id'] if best>.5 else None,'status':'pending'})
  if idx%50==0:print(idx,len(pool),flush=True)
 (OUT/'pool.json').write_text(json.dumps(pool,indent=2))
 # Select diverse source frames: up to 32 boundary targets and 16 low-overlap candidates.
 chosen=[];used=set()
 for group,budget in [(sorted([r for r in pool if r['edge'] and r['best_gt_iou']>.5],key=lambda r:r['source_id']),32),(sorted([r for r in pool if r['best_gt_iou']<=.5],key=lambda r:-r['detector_confidence']),16)]:
  unique=[]
  for r in group:
   if r['source_id'] not in used:unique.append(r);used.add(r['source_id'])
  indices=np.linspace(0,len(unique)-1,min(budget,len(unique)),dtype=int) if unique else []
  chosen.extend(unique[i] for i in indices)
 tiles=[]
 for i,r in enumerate(chosen):
  im=Image.open(DATA/r['source_image']).convert('RGB');x1,y1,x2,y2=r['bbox_xyxy'];box=[max(0,math.floor(x1)),max(0,math.floor(y1)),min(im.width,math.ceil(x2)),min(im.height,math.ceil(y2))]
  crop=im.crop(box);id=f'candidate_{i:03d}';path=OUT/'crops'/f'{id}.png';crop.save(path)
  r.update(id=id,crop=str(path.relative_to(OUT)),crop_sha256=sha(path),crop_integer_xyxy=box,evidence=f'page_{i//4+1:02d}.jpg')
  context=im.resize((480,270));d=ImageDraw.Draw(context);d.rectangle([v*.375 for v in box],outline='red',width=2)
  tile=Image.new('RGB',(760,300),'white');tile.paste(context,(0,30));thumb=ImageOps.contain(crop,(280,270));tile.paste(thumb,(480+(280-thumb.width)//2,30));d=ImageDraw.Draw(tile);d.text((5,5),f'{id} {r["source_id"]} proposal={r["proposed_class"]} IoU={r["best_gt_iou"]:.2f}',fill='black');tiles.append(tile)
 for i in range(0,len(tiles),4):
  page=Image.new('RGB',(760,300*len(tiles[i:i+4])),'white')
  for j,t in enumerate(tiles[i:i+4]):page.paste(t,(0,300*j))
  page.save(OUT/f'page_{i//4+1:02d}.jpg')
 (OUT/'candidates.json').write_text(json.dumps(chosen,indent=2));print('ready',len(chosen),flush=True)
if __name__=='__main__':main()
