"""Version a prompt-only experiment; preserve source labels and split assignments."""
import json
import argparse
import shutil
import yaml
from common import ROOT, sha, validate_manifest, validate_training_labels


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-sunflower',action='store_true',
                        help='Also replace class 2 with the top whole-pot potted prompt')
    args=parser.parse_args()
    experiment='flowers_clip_lily_chrysanthemum' if args.include_sunflower else 'flowers_clip_lily'
    src=ROOT/'train/datasets/flowers_video_holdout'
    dst=ROOT/'train/datasets'/f'{experiment}_v1'
    report_path=ROOT/'train/diagnostics/clip_flower_matching_v2/report.json'
    report=json.loads(report_path.read_text())
    ranking=report['summary']['1']['whole']['exact_potted_mean_ranking']
    assert ranking[0][0]=='lily'
    cfg_path=ROOT/'train/configs'/f'{experiment}.yaml'
    if args.include_sunflower:
        sunflower_ranking=report['summary']['2']['whole']['exact_potted_mean_ranking']
        if sunflower_ranking[0][0]!='chrysanthemum':
            raise SystemExit('CLIP ranking differs from the reviewed selection')
    if dst.exists() or cfg_path.exists():raise SystemExit('Experiment exists; refusing overwrite')
    source_hash=sha(src/'manifest.json')
    # Copy only source images/labels and manifest, not embeddings or dataset caches.
    dst.mkdir(parents=True)
    for name in ('images','labels'):shutil.copytree(src/name,dst/name)
    shutil.copyfile(src/'manifest.json',dst/'manifest.json')
    data=yaml.safe_load((src/'data.yaml').read_text());data['path']=str(dst)
    old_names=list(data['names']);data['names'][1]='potted lily'
    if args.include_sunflower:
        data['names'][2]='potted chrysanthemum'
    (dst/'data.yaml').write_text(yaml.safe_dump(data,sort_keys=False))
    rows,counts,_=validate_manifest(dst/'manifest.json',data['partition_policy'])
    validate_training_labels(rows,dst)
    for sub in ('images','labels'):
        for original in (src/sub).rglob('*'):
            if original.is_file():
                assert sha(original)==sha(dst/original.relative_to(src))
    assert sha(src/'manifest.json')==source_hash==sha(dst/'manifest.json')
    cfg=yaml.safe_load((ROOT/'train/configs/flowers.yaml').read_text())
    cfg.update(data=str(dst.relative_to(ROOT)),name=f'{experiment}_v1')
    cfg['data']+='/data.yaml'
    cfg['prompts']=data['names']
    # Display names are business names, not a botanical assertion from CLIP.
    cfg['display_names']=['芍药','兰花','向日葵','桉树叶']
    if args.include_sunflower:
        cfg['display_names']=['芍药','百合（原兰花）','菊花（原向日葵）','桉树叶']
    cfg_path.write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False))
    provenance={'result':'PASS','kind':'prompt_only_experiment','source_manifest_sha256':source_hash,
        'clip_report_sha256':sha(report_path),'old_prompts':old_names,'new_prompts':data['names'],
        'class_id_1_business_name':'兰花','class_id_1_training_prompt':'potted lily',
        'labels_changed':False,'new_annotations_accepted':False,'splits':counts,
        'notes':'Existing review statuses are historical provenance, not a fresh audit or CLIP-confirmed botanical identity. Known edge-label issues remain unchanged for controlled comparison.'}
    if args.include_sunflower:
        provenance.update(class_id_2_original_business_name='向日葵',
                          class_id_2_training_prompt='potted chrysanthemum',
                          selection_basis='mean cosine similarity, train crops, whole pot, exact potted prompt',
                          class_id_2_top_candidates=sunflower_ranking[:5],
                          display_names=cfg['display_names'])
    (dst/'prompt_provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2))
    print(json.dumps(provenance,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
