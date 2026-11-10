"""Optional four-flower cascade; standalone output, no service changes."""
import argparse,json
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image
from flower_classifier import encoder,features,crop_image,filter_candidates,ADAPTER,PROMPTS,DISPLAY,DEFAULT_OUT
from model_io import load_model
from common import ROOT,sha
from flower_visual_adapter import load_classifier

def main():
 p=argparse.ArgumentParser();p.add_argument('--classifier-dir',default=str(DEFAULT_OUT));p.add_argument('--operating-point',default=None);p.add_argument('--sources',nargs='+',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 folder=Path(a.classifier_dir);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
 payload=torch.load(folder/'classifier.pt',map_location='cpu',weights_only=True);assert payload['detector_adapter_sha256']==sha(ADAPTER)
 enc,pre=encoder();head=load_classifier(payload)
 point=json.loads((Path(a.operating_point) if a.operating_point else folder/'operating_point.json').read_text())
 if point.get('classifier_sha256') and point['classifier_sha256']!=sha(folder/'classifier.pt'):raise ValueError('Threshold configuration belongs to another classifier')
 det=load_model(ROOT/'models/yolov8s-worldv2.pt',ADAPTER);det.set_classes(PROMPTS)
 summary={'classifier_sha256':sha(folder/'classifier.pt'),'operating_point':point,'note':'Predictions only, not accuracy. Video has no audio.','inputs':[]}
 for idx,source in enumerate(a.sources):
  source=Path(source);isvideo=source.suffix.lower() in ('.mov','.mp4','.avi','.mkv');cap=cv2.VideoCapture(str(source)) if isvideo else None
  fps=cap.get(cv2.CAP_PROP_FPS) if cap else None;expected=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap else 1
  target=out/f'{idx+1:02d}_{source.stem}{".mp4" if cap else ".jpg"}';writer=None;n=0
  with (out/f'{idx+1:02d}_{source.stem}.jsonl').open('w') as log:
   try:
    while True:
     if cap:ok,frame=cap.read()
     else:frame=cv2.imread(str(source));ok=n==0 and frame is not None
     if not ok:break
     boxes=det.predict(frame,device='cpu',imgsz=640,conf=min(point['detector_conf']) if isinstance(point['detector_conf'],list) else point['detector_conf'],iou=.7,verbose=False)[0].boxes.data.cpu().tolist()
     rgb=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));f=features(enc,pre,[crop_image(rgb,b[:4]) for b in boxes])
     with torch.no_grad():scores=head(f).softmax(1).tolist()
     candidates=[]
     for b,s in zip(boxes,scores):
      c=int(np.argmax(s));candidates.append({'class':c,'xyxy':b[:4],'confidence':b[4]*s[c],'detector_confidence':b[4],'classifier_confidence':s[c]})
     keep=filter_candidates(candidates,point['detector_conf'],point['classifier_conf'],point['nms_iou'])
     font_scale=max(.6,frame.shape[1]/1600);line_width=max(2,round(font_scale*2))
     for r in keep:
      xy=[round(x) for x in r['xyxy']];cv2.rectangle(frame,tuple(xy[:2]),tuple(xy[2:]),(0,220,255),line_width);cv2.putText(frame,f"{DISPLAY[r['class']]} {r['confidence']:.2f}",(max(0,xy[0]),max(round(30*font_scale),xy[1]-8)),0,font_scale,(0,80,240),line_width)
     log.write(json.dumps({'frame':n,'detections':keep})+'\n')
     if cap:
      if writer is None:
       h,w=frame.shape[:2];writer=cv2.VideoWriter(str(target),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h));assert writer.isOpened()
      writer.write(frame)
     else:assert cv2.imwrite(str(target),frame)
     n+=1
     if n%100==0:print(source.name,n,expected,flush=True)
   finally:
    if cap:cap.release()
    if writer:writer.release()
  assert n==expected and n>0
  summary['inputs'].append({'source':str(source),'source_sha256':sha(source),'frames':n,'output':str(target)})
 (out/'summary.json').write_text(json.dumps(summary,indent=2))
if __name__=='__main__':main()
