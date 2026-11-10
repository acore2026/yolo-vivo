import sys,json,hashlib,cv2
from pathlib import Path
from PIL import Image,ImageDraw
root=Path('/home/huawei/VIVO/yolo-vivo'); out=root/'train/diagnostics/flower_hard_batch_v2'
rows=json.loads((root/'train/datasets/flowers_reviewed_v2_trainval/manifest.json').read_text()); new=[]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(out/'images').mkdir(exist_ok=True)
for video,count in [(1,12),(4,2),(5,2),(6,2),(7,2)]:
 old=sorted([r for r in rows if r['source']==f'data/flower{video}.mp4'],key=lambda r:r['frame']); source=root/old[0]['source']; assert sha(source)==old[0]['source_sha256']
 cap=cv2.VideoCapture(str(source)); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=cap.get(cv2.CAP_PROP_FPS)
 for i in range(count):
  target=(i+.5)*n/count
  gaps=[(a['frame']+b['frame'])//2 for a,b in zip(old,old[1:])]; frame=min(gaps,key=lambda f:abs(f-target)); assert frame not in [r['frame'] for r in old]
  cap.set(cv2.CAP_PROP_POS_FRAMES,frame); ok,img=cap.read(); assert ok
  ident=f'flower{video}_f{frame:06d}'; path=out/'images'/f'{ident}.png'; assert not path.exists(); cv2.imwrite(str(path),img)
  new.append(dict(id=ident,source=old[0]['source'],source_sha256=old[0]['source_sha256'],frame=frame,timestamp_seconds=frame/fps,source_fps=fps,source_total_frames=n,width=img.shape[1],height=img.shape[0],image_sha256=sha(path),status='pending',group='flower_same_setup_01',split='train'))
 cap.release()
(out/'pending_manifest.json').write_text(json.dumps(new,ensure_ascii=False,indent=2))
for k in range(0,len(new),4):
 sheet=Image.new('RGB',(1280,776),'#202020'); d=ImageDraw.Draw(sheet)
 for j,r in enumerate(new[k:k+4]):
  x=(j%2)*640;y=(j//2)*388; im=Image.open(out/'images'/f'{r["id"]}.png');im.thumbnail((640,360));sheet.paste(im,(x,y+28));d.text((x+6,y+6),r['id'],fill='white')
 sheet.save(out/f'raw_{k//4+1:02d}.jpg',quality=95)
print(json.dumps(new,indent=2))
