"""Train-only resolution/context/perspective augmentation, linear vs feature adapter.
Never reads test images, labels or predictions. One fixed seed; validation is development only.
"""
import json,random,copy
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image,ImageOps,ImageDraw,ImageEnhance
from flower_classifier import DATA,ROOT,ADAPTER,PROMPTS,DEFAULT_OUT,encoder,features,crop_image,class_metrics,macrof1,eval_detection,CLIP_SHA256
from flower_visual_adapter import VisualFeatureAdapter,load_classifier
from common import sha
OUT=ROOT/'train/runs/flower_classifier_v4'

def augment(im,box,variant,rng):
    if variant=='context':
        result=crop_image(im,box,pad=.18)
        return result,{'context_padding':.18,'note':'real same-image context; no pasted flowers'}
    result=crop_image(im,box)
    if variant=='low_resolution':
        side=rng.choice([48,64,96]);result=result.resize((side,side),Image.Resampling.BILINEAR).resize((224,224),Image.Resampling.BILINEAR)
        gain=rng.uniform(.7,1.1);result=ImageEnhance.Brightness(result).enhance(gain)
        return result,{'intermediate_size':side,'brightness':gain}
    # Inward trapezoid with all corners retained; limited 2-D distortion, not a new 3-D view.
    result=result.resize((224,224),Image.Resampling.BILINEAR);inset=rng.uniform(.03,.12)*223
    src=np.float32([[0,0],[223,0],[223,223],[0,223]])
    dst=np.float32([[inset,0],[223-inset,0],[223,223],[0,223]])
    mat=cv2.getPerspectiveTransform(src,dst);array=cv2.warpPerspective(np.array(result),mat,(224,224),borderValue=(127,127,127))
    angle=rng.uniform(-12,12);result=Image.fromarray(array).rotate(angle,expand=True,fillcolor=(127,127,127))
    return result,{'perspective_matrix':mat.tolist(),'rotation_degrees':angle,'note':'2D simulation only, not true overhead photography'}

def main():
    OUT.mkdir(exist_ok=False);torch.set_num_threads(4);torch.manual_seed(42);rng=random.Random(42)
    original=torch.load(DEFAULT_OUT/'features.pt',map_location='cpu',weights_only=True)
    extra_path=ROOT/'train/runs/flower_classifier_v3/extra_features.pt'
    extra=torch.load(extra_path,map_location='cpu',weights_only=True)
    assert original['source_manifest_sha256']==sha(DATA/'manifest.json')
    assert extra['original_features_sha256']==sha(DEFAULT_OUT/'features.pt')
    enc,pre=encoder();images=[];ys=[];provenance=[];preview=[]
    rows=json.loads((DATA/'manifest.json').read_text())
    for r in rows:
        if r['split']!='train':continue
        assert sha(DATA/r['image'])==r['image_sha256']
        with Image.open(DATA/r['image']) as im:
            im=im.convert('RGB')
            for j,obj in enumerate(r['objects']):
                for variant in ['context','low_resolution','perspective']:
                    aug,settings=augment(im,obj['bbox_xyxy'],variant,rng)
                    images.append(aug);ys.append(obj['class_id'])
                    provenance.append({'source_id':r['id'],'source_image_sha256':r['image_sha256'],'object_index':j,'class_id':obj['class_id'],'split':'train','variant':variant,'bbox_xyxy':obj['bbox_xyxy'],'settings':settings,'review':'augmentation from reviewed annotation; not individually accepted'})
                    # Deterministic first 2 distinct source objects per class and variant.
                    if sum(x['class_id']==obj['class_id'] and x['variant']==variant for x in preview)<2:
                        preview.append({'class_id':obj['class_id'],'variant':variant,'source_id':r['id'],'image':aug.copy(),'index':len(provenance)-1})
    for start in range(0,len(preview),8):
        page=Image.new('RGB',(1120,600),'white')
        for idx,r in enumerate(preview[start:start+8]):
            tile=Image.new('RGB',(280,300),'white');thumb=ImageOps.contain(r['image'],(280,250));tile.paste(thumb,((280-thumb.width)//2,0));d=ImageDraw.Draw(tile);d.text((4,255),f"class {r['class_id']} {r['variant']}",fill='black');d.text((4,275),r['source_id'],fill='black');page.paste(tile,((idx%4)*280,(idx//4)*300))
        page.save(OUT/f'augmentation_{start//8+1:02d}.jpg')
    (OUT/'augmentation_preview_index.json').write_text(json.dumps([{k:v for k,v in r.items() if k!='image'} for r in preview],indent=2))
    added=features(enc,pre,images);print('Added train-only variants',len(ys),flush=True)
    x=torch.cat([original['features']['train'],extra['extra_features'],added]);y=torch.cat([original['labels']['train'],extra['extra_labels'],torch.tensor(ys)])
    assert torch.isfinite(x).all();assert len(x)==len(y)
    torch.save({'added_features':added,'added_labels':torch.tensor(ys),'provenance':provenance,'original_features_sha256':sha(DEFAULT_OUT/'features.pt'),'v3_extra_features_sha256':sha(extra_path)},OUT/'augmentation_features.pt')
    # Encode unchanged val detector proposals once for both controlled variants.
    candidates_path=DEFAULT_OUT/'candidate_recovery/val_candidates.json'
    candidates=json.loads(candidates_path.read_text());cf=[];lengths=[]
    for r in candidates:
        assert sha(DATA/r['image'])==r['image_sha256']
        with Image.open(DATA/r['image']) as im:ff=features(enc,pre,[crop_image(im.convert('RGB'),b['xyxy']) for b in r['candidates']])
        lengths.append(len(ff));cf.append(ff)
    candidate_features=torch.cat(cf)
    torch.save({'features':candidate_features,'lengths':lengths,'source_sha256':sha(candidates_path)},OUT/'val_candidate_features.pt')
    del enc
    weights=1/torch.bincount(y).float();weights/=weights.mean();weights[4]*=.25
    # v3 threshold frozen before either v4 trial; no additional threshold search here.
    fixed_path=ROOT/'train/runs/flower_classifier_v3/per_class_calibration/operating_point.json'
    fixed=json.loads(fixed_path.read_text());allreports={}
    for kind in ['linear','visual_feature_adapter']:
        target=OUT/kind;target.mkdir();torch.manual_seed(42)
        head=torch.nn.Linear(512,5) if kind=='linear' else VisualFeatureAdapter()
        lr=.02 if kind=='linear' else .002
        optimizer=torch.optim.AdamW(head.parameters(),lr=lr,weight_decay=.01)
        best=(-1,float('-inf'));history=[]
        for epoch in range(1,301):
            head.train();optimizer.zero_grad();loss=torch.nn.functional.cross_entropy(head(x),y,weight=weights);loss.backward();optimizer.step()
            head.eval()
            with torch.no_grad():
                v=head(original['features']['val']);vl=float(torch.nn.functional.cross_entropy(v,original['labels']['val']));stats=class_metrics(original['labels']['val'],v.argmax(1));score=macrof1(stats)
            history.append({'epoch':epoch,'train_loss':float(loss.detach()),'val_loss':vl,'val_crop_macro_f1':score})
            if (score,-vl)>best:best=(score,-vl);state=copy.deepcopy(head.state_dict());bestepoch=epoch
        head.load_state_dict(state);head.eval()
        with torch.no_grad():
            crop_stats=class_metrics(original['labels']['val'],head(original['features']['val']).argmax(1));scores=head(candidate_features).softmax(1).tolist()
        records=copy.deepcopy(candidates);idx=0
        for r in records:
            for b in r['candidates']:
                s=scores[idx];idx+=1;c=int(np.argmax(s));b.update({'class':c,'classifier_confidence':s[c],'confidence':s[c]*b['detector_confidence'],'class_probabilities':s})
        stats=eval_detection(records,fixed['detector_conf'],fixed['classifier_conf'])
        payload={'format_version':3,'head_kind':kind,'head':state,'num_classes':5,'bottleneck':64,'encoder_sha256':CLIP_SHA256,'detector_adapter_sha256':sha(ADAPTER),'prompts':PROMPTS,'reject_class':4,'seed':42,'best_epoch':bestepoch,'source_manifest_sha256':sha(DATA/'manifest.json'),'review_manifest_sha256':sha(ROOT/'train/diagnostics/flower_combined_review_v3/reviewed.json'),'augmentation_sha256':sha(OUT/'augmentation_features.pt'),'training_counts':torch.bincount(y).tolist(),'encoder_frozen':True,'optimizer':{'lr':lr,'weight_decay':.01,'epochs':300,'reject_loss_multiplier':.25}}
        torch.save(payload,target/'classifier.pt')
        loaded=load_classifier(torch.load(target/'classifier.pt',map_location='cpu',weights_only=True))
        with torch.no_grad():assert torch.allclose(loaded(candidate_features),head(candidate_features),atol=1e-6)
        point={'detector_conf':fixed['detector_conf'],'classifier_conf':fixed['classifier_conf'],'nms_iou':fixed['nms_iou'],'stats':stats,'macro_f1':macrof1(stats),'passes':all(s['precision']>.9 and s['recall']>.9 for s in stats),'classifier_sha256':sha(target/'classifier.pt'),'scope':'same v3 val thresholds, no new grid search; val epoch selection'}
        report={'kind':kind,'gt_crop_val':crop_stats,'end_to_end_val':point,'best_epoch':bestepoch,'train_count':len(y),'train_class_counts':torch.bincount(y).tolist(),'test_used':False,'visual_encoder_backbone_finetuned':False,'scope':'supervised feature adaptation only; not new independent scene evidence','threshold_source_sha256':sha(fixed_path)}
        for name,data in [('report.json',report),('history.json',history),('val_candidates.json',records),('operating_point.json',point)]: (target/name).write_text(json.dumps(data,indent=2))
        allreports[kind]=report;print(kind,json.dumps(point),flush=True)
    (OUT/'comparison.json').write_text(json.dumps(allreports,indent=2))
if __name__=='__main__':main()
