"""Manually drawn test2 three-class sample boxes; frozen before new-model tests."""
import json,cv2
from pathlib import Path
from PIL import Image,ImageDraw
from common import ROOT,sha
from three_flower_data import dump,PROMPTS,NAMES
FRAMES=[0,83,167,250,333,416,500,583,666,749,833,916]
# Coordinates independently placed on the 640x360 raw review images, whole pot.
BOXES=[
[(0,75,121,225,243),(1,296,108,395,215),(2,388,112,478,210)],
[(0,24,111,198,261),(1,293,96,412,225),(2,397,97,509,217)],
[(0,24,145,264,350),(1,355,99,514,280),(2,504,107,640,262)],
[(0,0,141,174,312),(1,291,105,434,256),(2,419,110,536,244)],
[(0,0,176,97,317),(1,204,145,339,268),(2,322,153,421,253)],
[(1,61,121,334,360),(2,295,127,512,340)],
[(0,30,173,212,324),(1,282,119,416,260),(2,400,132,497,239)],
[(0,31,167,230,327),(1,309,130,418,256),(2,406,134,504,241)],
[(0,18,174,193,312),(1,264,133,381,246),(2,368,143,460,234)],
[(0,69,132,286,315),(1,363,85,495,239),(2,475,109,573,219)],
[(0,0,159,172,331),(1,265,111,430,279),(2,408,128,523,257)],
[(0,17,182,179,312),(1,289,144,398,255),(2,381,146,481,242)]]
def main():
 out=ROOT/'train/datasets/flowers_three_test2_samples';out.mkdir(parents=True,exist_ok=False);(out/'images/test').mkdir(parents=True);(out/'labels/test').mkdir(parents=True);(out/'review').mkdir();cap=cv2.VideoCapture(str(ROOT/'test_videos/test2.MOV'));digest=sha(ROOT/'test_videos/test2.MOV');rows=[]
 for f,boxes in zip(FRAMES,BOXES):
  cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,im=cap.read();assert ok;ident=f'test2_f{f:06d}';img=out/'images/test'/f'{ident}.png';cv2.imwrite(str(img),im);label=out/'labels/test'/f'{ident}.txt';label.write_text(''.join(f'{c} {(x1+x2)/1280:.9f} {(y1+y2)/720:.9f} {(x2-x1)/640:.9f} {(y2-y1)/360:.9f}\n' for c,x1,y1,x2,y2 in boxes));tile=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)).resize((640,360));d=ImageDraw.Draw(tile)
  for c,x1,y1,x2,y2 in boxes:d.rectangle((x1,y1,x2,y2),outline=['red','magenta','yellow'][c],width=2);d.text((x1+3,y1+3),str(c),fill='white',stroke_width=1,stroke_fill='black')
  tile.save(out/'review'/f'{ident}.jpg',quality=95);rows.append(dict(id=ident,source='test_videos/test2.MOV',source_sha256=digest,frame=f,width=1920,height=1080,image=str(img.relative_to(out)),image_sha256=sha(img),label=str(label.relative_to(out)),label_sha256=sha(label),gt=[{'class':c,'xyxy':[x1*3,y1*3,x2*3,y2*3]} for c,x1,y1,x2,y2 in boxes],status='pending_overlay_review',human_review=False))
 cap.release();dump(out/'annotations_pending.json',rows)
 for start in range(0,12,4):
  sheet=Image.new('RGB',(1280,776),'#202020');d=ImageDraw.Draw(sheet)
  for j,r in enumerate(rows[start:start+4]):
   x=j%2*640;y=j//2*388;sheet.paste(Image.open(out/'review'/f'{r["id"]}.jpg'),(x,y+28));d.text((x+5,y+5),r['id'],fill='white')
  sheet.save(out/'review'/f'page_{start//4+1:02d}.jpg',quality=95)
if __name__=='__main__':main()
