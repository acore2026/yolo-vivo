"""Render retained predictions and animated flower introductions on original video."""
import argparse,json,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
from service.flower_overlay import FlowerOverlays,INTRODUCTIONS

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--three-classes',action='store_true');p.add_argument('--source',type=Path,required=True);p.add_argument('--predictions-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 summary=json.loads((a.predictions_dir/'summary.json').read_text());source_index=next(i for i,r in enumerate(summary['inputs']) if r['source_sha256']==digest(a.source))
 predictions=a.predictions_dir/f'{source_index+1:02d}_{a.source.stem}.jsonl';rows=[json.loads(s) for s in predictions.read_text().splitlines()]
 cap=cv2.VideoCapture(str(a.source));assert cap.isOpened();fps=cap.get(cv2.CAP_PROP_FPS);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));assert len(rows)==n
 a.output.mkdir(parents=True,exist_ok=False);writer=cv2.VideoWriter(str(a.output/'preview.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h));assert writer.isOpened();overlay=FlowerOverlays();samples={30,90,180,300,450,600,750,900};names=['peony','orchid','sunflower','eucalyptus']
 introductions=INTRODUCTIONS
 if a.three_classes:
  from service.three_flower_overlay import ThreeFlowerOverlays,INTRODUCTIONS as THREE_INTRODUCTIONS
  overlay=ThreeFlowerOverlays();names=['peony','lily','chrysanthemum'];introductions=THREE_INTRODUCTIONS
 with (a.output/'animation.jsonl').open('x') as log:
  try:
   for i,row in enumerate(rows):
    ok,frame=cap.read();assert ok and row['frame']==i;now=i/fps;kept=overlay.update(row['detections'],now)
    for d in kept:
     x1,y1,x2,y2=map(round,d['xyxy']);cv2.rectangle(frame,(x1,y1),(x2,y2),(90,225,240),2);cv2.putText(frame,f"{names[d['class']]} {d['confidence']:.2f}",(max(0,x1),max(22,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.7,(90,225,240),2)
    overlay.draw(frame,now);writer.write(frame)
    log.write(json.dumps(dict(frame=i,tracks=[dict(id=k,class_id=t['class'],visible=t['overlay'].arrived,missing=t['overlay'].missing_frames,characters=len(t['overlay'].text_at(now) or '')) for k,t in overlay.tracks.items()]))+'\n')
    if i in samples:assert cv2.imwrite(str(a.output/f'preview_{i:06d}.jpg'),frame)
    if i%200==0:print(i,n,flush=True)
  finally:cap.release();writer.release()
 metadata=dict(source=str(a.source),source_sha256=digest(a.source),prediction_sha256=digest(predictions),model=summary,frames=n,fps=fps,introductions=introductions,style=dict(width=220,font=14,heading_font=18,character_seconds=.04,playback='type_once_then_hold_until_lost',required_frames=10,anchor='right of individual box, clip at edge'),notes=['Uses video time, not rendering wall clock','Lightweight same-class IoU tracking; occlusion or class changes can reset identity','Presentation-only same-class duplicate suppression at IoU >= 0.5','Panels may overlap in crowded scenes; fixed right anchor retained','No model training or live-camera service replacement'])
 (a.output/'summary.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
