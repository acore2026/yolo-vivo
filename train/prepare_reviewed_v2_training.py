"""Create a standalone train/val input from reviewed v2; never promote test reviews."""
import json
import shutil
import yaml
from common import ROOT, sha, validate_manifest, validate_training_labels


def main():
    source = ROOT/'train/datasets/flowers_reviewed_v2'
    target = ROOT/'train/datasets/flowers_reviewed_v2_trainval'
    config_path = ROOT/'train/configs/flowers_reviewed_v2.yaml'
    rows = json.loads((source/'manifest.json').read_text())
    selected = [r for r in rows if r['split'] in ('train', 'val')]
    assert len(selected) == 319
    assert all(r['status'] == 'accepted' and r['review']['reviewer'] == 'assistant_visual' for r in selected)
    if target.exists() or config_path.exists():
        raise SystemExit('Output exists; refusing to overwrite')
    for r in selected:
        for key in ('image', 'label'):
            assert sha(source/r[key]) == r[key+'_sha256'], r['id']
    target.mkdir(parents=True)
    for r in selected:
        for key in ('image', 'label'):
            dest = target/r[key]
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source/r[key], dest)
            assert sha(dest) == r[key+'_sha256']
    (target/'manifest.json').write_text(json.dumps(selected, ensure_ascii=False, indent=2))
    data = yaml.safe_load((source/'data.yaml').read_text())
    data.pop('test', None)
    data['path'] = str(target)
    (target/'data.yaml').write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
    checked, counts, _ = validate_manifest(target/'manifest.json', 'same_setup_video_holdout')
    validate_training_labels(checked, target)
    assert counts == {'train': 280, 'val': 39}
    cfg = yaml.safe_load((ROOT/'train/configs/flowers_clip_lily_chrysanthemum.yaml').read_text())
    cfg['data'] = str((target/'data.yaml').relative_to(ROOT))
    cfg['name'] = 'flowers_reviewed_v2'
    config_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    provenance = {'source': str(source), 'source_manifest_sha256': sha(source/'manifest.json'),
                  'manifest_sha256': sha(target/'manifest.json'), 'counts': counts,
                  'copied_image_label_hashes': 'PASS', 'preflight': 'PASS',
                  'test_excluded': 80, 'training_settings_changed': False,
                  'independent_scene_evaluation': False}
    (target/'preparation_report.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2))
    print(json.dumps(provenance, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
