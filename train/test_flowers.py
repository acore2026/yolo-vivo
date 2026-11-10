"""Detect four potted flowers in local images/videos, exporting overlays and JSON."""
import argparse
from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path
import time
import cv2
from common import resolve, config, sha

DEFAULT_ADAPTER='train/diagnostics/adapter_best_architecture_v2.pt'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources',nargs='+',default=['test_videos/test.JPG','test_videos/test.MOV'])
    p.add_argument('--adapter',default=DEFAULT_ADAPTER)
    p.add_argument('--config',default='train/configs/flowers.yaml')
    p.add_argument('--output',default=None)
    p.add_argument('--device',default='auto')
    p.add_argument('--conf',type=float,default=0.25)
    p.add_argument('--iou',type=float,default=0.7)
    p.add_argument('--imgsz',type=int,default=640)
    a=p.parse_args()
    if not 0<a.conf<1 or not 0<a.iou<=1: p.error('Invalid confidence/IoU')
    sources=[resolve(s) for s in a.sources]
    if not all(s.is_file() for s in sources): p.error('Missing input file')
    import torch
    torch.set_num_threads(4)
    device=('0' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device
    from model_io import load_model
    c=config(a.config);adapter=resolve(a.adapter)
    model=load_model(resolve(c['model']),adapter);model.set_classes(c['prompts'])
    out=resolve(a.output or 'train/predictions/flowers_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True,exist_ok=False)
    summary={'device':device,'adapter':str(adapter),'adapter_sha256':sha(adapter),'base_sha256':sha(resolve(c['model'])),
             'conf':a.conf,'iou':a.iou,'imgsz':a.imgsz,'classes':c['display_names'],
             'note':'Detection counts are predictions, not precision/recall; video output has no audio. All frames processed.','inputs':[]}
    start=time.monotonic()
    for index,source in enumerate(sources):
        stem=f'{index+1:02d}_{source.stem}'
        is_video=source.suffix.lower() in {'.mov','.mp4','.avi','.mkv'}
        cap=cv2.VideoCapture(str(source)) if is_video else None
        fps=cap.get(cv2.CAP_PROP_FPS) if cap else None
        if cap and (not cap.isOpened() or not math.isfinite(fps) or fps<=0): raise RuntimeError('Invalid video')
        total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap else 1
        writer=None;count=0;boxes=Counter();frames=Counter();best={};image_rows=[]
        target=out/(stem+'.mp4' if is_video else stem+'.jpg')
        with (out/(stem+'.jsonl')).open('w') as log:
            try:
                while True:
                    if cap: ok,frame=cap.read()
                    else:
                        frame=cv2.imread(str(source));ok=frame is not None and count==0
                    if not ok:
                        if count==0: raise RuntimeError(f'Cannot decode {source}')
                        break
                    result=model.predict(frame,device=device,imgsz=a.imgsz,conf=a.conf,iou=a.iou,verbose=False)[0]
                    detections=[];present=set()
                    for box in result.boxes:
                        cls=int(box.cls.item());score=float(box.conf.item());present.add(cls);boxes[cls]+=1
                        detections.append({'class_id':cls,'class_name':c['display_names'][cls],'confidence':score,'xyxy':box.xyxy[0].tolist()})
                    frames.update(present)
                    plotted=result.plot()
                    if cap:
                        if writer is None:
                            h,w=plotted.shape[:2];writer=cv2.VideoWriter(str(target),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
                            if not writer.isOpened(): raise RuntimeError('Video writer failed')
                        writer.write(plotted)
                        for cls in present:
                            score=max(d['confidence'] for d in detections if d['class_id']==cls)
                            if score>best.get(cls,{}).get('confidence',-1):
                                best[cls]={'frame':count,'confidence':score,'preview':f'{stem}_class{cls}_best.jpg'}
                                if not cv2.imwrite(str(out/best[cls]['preview']),plotted): raise RuntimeError('Preview write failed')
                    else:
                        if not cv2.imwrite(str(target),plotted): raise RuntimeError('Image write failed')
                        image_rows=detections
                    log.write(json.dumps({'frame':count,'seconds':count/fps if fps else None,'detections':detections},ensure_ascii=False)+'\n')
                    count+=1
                    if count%100==0: print(f'{source.name}: {count}/{total} frames',flush=True)
                    if not cap: break
            finally:
                if writer: writer.release()
                if cap: cap.release()
        if is_video and count!=total: raise RuntimeError(f'Video decode incomplete: {count}/{total}')
        summary['inputs'].append({'source':str(source),'source_sha256':sha(source),'output':str(target),'frames':count,'fps':fps,
           'boxes_by_class':{n:boxes[i] for i,n in enumerate(c['display_names'])},
           'frames_with_class':{n:frames[i] for i,n in enumerate(c['display_names'])},'best_previews':best,'image_detections':image_rows})
        summary['elapsed_seconds']=time.monotonic()-start
        (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
