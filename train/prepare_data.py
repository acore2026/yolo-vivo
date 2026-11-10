"""Validate approved group assignments, optionally export a new training dataset."""
import argparse
import json
import shutil
import yaml
from common import resolve, validate_manifest, validate_training_labels, sha

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',default='work/flower_annotation_20261103_400/manifest.json')
    p.add_argument('--groups',help='JSON mapping approved capture group IDs to train/val/test')
    p.add_argument('--video-plan',help='Approved whole-video proposal; explicitly allows same-setup holdout')
    p.add_argument('--output',help='New output directory; omitted means read-only validation')
    a=p.parse_args()
    rows,counts,groups=validate_manifest(a.manifest)
    print(json.dumps({'accepted_by_split':counts,'groups':{k:sorted(v) for k,v in groups.items()}},ensure_ascii=False))
    if not a.output: return
    if bool(a.groups)==bool(a.video_plan): raise SystemExit('Specify exactly one of --groups or --video-plan')
    policy='same_setup_video_holdout' if a.video_plan else 'capture_group'
    key='source' if a.video_plan else 'group'
    if a.video_plan:
        plan=json.loads(resolve(a.video_plan).read_text())
        if plan['manifest_sha256']!=sha(resolve(a.manifest)): raise SystemExit('Approved manifest has changed')
        assignment=plan['video_assignments']
        expected={r['id']:(r['source'],r['frame'],r['image_sha256'],assignment[r['source']]) for r in rows if r['status']=='accepted'}
        proposed={r['id']:(r['source'],r['frame'],r['image_sha256'],r['proposed_split']) for r in plan['items']}
        if expected!=proposed or len(plan['items'])!=len(expected): raise SystemExit('Plan items differ from manifest')
    else:
        assignment=json.loads(resolve(a.groups).read_text())
    if set(assignment)!={r[key] for r in rows if r['status']=='accepted'} or not {'train','val'}<=set(assignment.values()) or not set(assignment.values())<={'train','val','test'}:
        raise SystemExit('Require all groups assigned once; independent train and val groups required.')
    out=resolve(a.output)
    out.mkdir(parents=True,exist_ok=False)
    origin=resolve(a.manifest).parent
    exported=[]
    for r in rows:
        if r['status']!='accepted': continue
        split=assignment[r[key]]
        image=out/'images'/split/(r['id']+'.png');label=out/'labels'/split/(r['id']+'.txt')
        image.parent.mkdir(parents=True,exist_ok=True);label.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(origin/r['image'],image)
        lines=[]
        for o in r['objects']:
            x1,y1,x2,y2=o['bbox_xyxy'];w,h=r['width'],r['height']
            lines.append(f"{o['class_id']} {(x1+x2)/2/w:.9f} {(y1+y2)/2/h:.9f} {(x2-x1)/w:.9f} {(y2-y1)/h:.9f}")
        label.write_text('\n'.join(lines)+'\n')
        record={**r,'split':split,'image':str(image.relative_to(out)), 'partition_unit':r[key]}
        for old in ('export_image','export_label','export_label_sha256','preview'):
            record.pop(old,None)
        record['label']=str(label.relative_to(out));record['label_sha256']=sha(label)
        exported.append(record)
    names=['potted peony','potted orchid','potted sunflower','potted eucalyptus']
    data={'path':str(out),'names':names,'partition_policy':policy,**{s:f'images/{s}' for s in sorted(set(assignment.values()))}}
    (out/'data.yaml').write_text(yaml.safe_dump(data,sort_keys=False))
    (out/'manifest.json').write_text(json.dumps(exported,ensure_ascii=False,indent=2))
    checked,counts,_=validate_manifest(out/'manifest.json',policy)
    validate_training_labels(checked,out)
    report={'result':'PASS','partition_policy':policy,'images':counts,'source_manifest_sha256':sha(resolve(a.manifest)),
            'capture_group_preserved':True,'augmentation_performed':False,'checks':['source image hashes','label coordinate roundtrip','image-label pairing','source video isolation','identical image isolation']}
    (out/'validation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
