"""Original frozen OpenAI CLIP text matching on train-only annotated crops."""
import json
import argparse
from collections import defaultdict, Counter
import numpy as np
import torch
from PIL import Image, ImageOps, ImageDraw
from common import ROOT, sha

CANDIDATES=['peony','orchid','lotus','water lily','lily','tulip','rose','sunflower',
'daisy','chrysanthemum','gerbera','carnation','dahlia','hydrangea','magnolia','camellia',
'azalea','hibiscus','poppy','marigold','daffodil','iris','begonia','anthurium',
'eucalyptus','olive','boxwood','fern','succulent','lavender','cherry blossom','rhododendron']
TEMPLATES=['a photo of {name} in a flower pot.',
           'a photo of artificial {name} in a flower pot.',
           'a photo of a {name} arrangement.',
           'a photo of {name}.']

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='train/diagnostics/clip_flower_matching_v2')
    args=parser.parse_args()
    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=False)
    from runtime import setup, CLIP_SHA256
    setup()
    import clip
    weights=ROOT/'train/.cache/weights/clip/ViT-B-32.pt'
    assert sha(weights)==CLIP_SHA256
    torch.set_num_threads(4)
    model,preprocess=clip.load(str(weights),device='cpu',jit=False);model.eval()
    root=ROOT/'train/datasets/flowers_video_holdout'
    rows=json.loads((root/'manifest.json').read_text())
    groups=defaultdict(list)
    for row in rows:
        if row['split']!='train':continue
        for j,obj in enumerate(row['objects']):groups[(obj['class_id'],row['source'])].append((row,j,obj))
    selected=[]
    for key,items in sorted(groups.items()):
        selected.extend(items[i] for i in np.unique(np.linspace(0,len(items)-1,min(8,len(items)),dtype=int)))
    with torch.inference_mode():
        text=model.encode_text(clip.tokenize([t.format(name=c) for c in CANDIDATES for t in TEMPLATES]))
        text=text/text.norm(dim=-1,keepdim=True)
        text=text.reshape(len(CANDIDATES),len(TEMPLATES),-1)
        ensemble=text.mean(1);ensemble=ensemble/ensemble.norm(dim=-1,keepdim=True)
        exact=model.encode_text(clip.tokenize(['potted '+c for c in CANDIDATES]))
        exact=exact/exact.norm(dim=-1,keepdim=True)
        records=[];tiles=[]
        for index,(row,j,obj) in enumerate(selected):
            image=Image.open(root/row['image']).convert('RGB')
            crop=image.crop(tuple(round(x) for x in obj['bbox_xyxy']))
            # Retain the complete annotated object via square padding, plus a flower-focused upper crop.
            variants={'whole':crop,'upper':crop.crop((0,0,crop.width,max(1,round(crop.height*.7))))}
            record={'id':row['id'],'class_id':obj['class_id'],'source':row['source'],'frame':row['frame'],
                    'image_sha256':row['image_sha256'],'object_index':j,'bbox_xyxy':obj['bbox_xyxy'],'variants':{}}
            for name,img in variants.items():
                square=Image.new('RGB',(max(img.size),)*2,(127,127,127));square.paste(img,((square.width-img.width)//2,(square.height-img.height)//2))
                feature=model.encode_image(preprocess(square).unsqueeze(0));feature=feature/feature.norm(dim=-1,keepdim=True)
                scores=(feature@ensemble.T)[0].tolist()
                by_template=torch.einsum('bd,ctd->bct',feature,text)[0]
                record['variants'][name]={'cosine_scores':dict(zip(CANDIDATES,scores)),
                  'exact_potted_scores':dict(zip(CANDIDATES,(feature@exact.T)[0].tolist())),
                  'top5':sorted(zip(CANDIDATES,scores),key=lambda x:-x[1])[:5],
                  'template_winners':[CANDIDATES[int(i)] for i in by_template.argmax(0)]}
            tile=Image.new('RGB',(240,280),'white');tile.paste(ImageOps.contain(crop,(240,230)),(0,0))
            draw=ImageDraw.Draw(tile);draw.text((3,232),f"{index+1:02d} class {obj['class_id']} {row['id']}",fill='black')
            draw.text((3,248),record['variants']['whole']['top5'][0][0],fill='black')
            tiles.append(tile);records.append(record)
    summary={}
    for cid in range(4):
        group=[r for r in records if r['class_id']==cid];summary[str(cid)]={}
        for variant in ('whole','upper'):
            means={c:float(np.mean([r['variants'][variant]['cosine_scores'][c] for r in group])) for c in CANDIDATES}
            summary[str(cid)][variant]={'n':len(group),'mean_ranking':sorted(means.items(),key=lambda x:-x[1]),
              'exact_potted_mean_ranking':sorted([(c,float(np.mean([r['variants'][variant]['exact_potted_scores'][c] for r in group]))) for c in CANDIDATES],key=lambda x:-x[1]),
              'top1_votes':dict(Counter(r['variants'][variant]['top5'][0][0] for r in group))}
    for start in range(0,len(tiles),16):
        sheet=Image.new('RGB',(960,1120),'white')
        for i,tile in enumerate(tiles[start:start+16]):sheet.paste(tile,((i%4)*240,(i//4)*280))
        sheet.save(out/f'crops_{start//16+1:02d}.jpg')
    report={'clip':'OpenAI CLIP ViT-B/32 original weights','weights_sha256':sha(weights),'device':'cpu',
            'selection':'train only; up to 8 evenly spaced annotated objects per class/source video',
            'templates':TEMPLATES,'candidates':CANDIDATES,'summary':summary,'samples':records,
            'note':'Cosine similarity is not probability or botanical ground truth. No val/test crop used.'}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    for cid,s in summary.items():
        print(cid,{v:{'top5':r['mean_ranking'][:5],'exact_potted_top5':r['exact_potted_mean_ranking'][:5],'votes':r['top1_votes']} for v,r in s.items()},flush=True)

if __name__=='__main__':main()
