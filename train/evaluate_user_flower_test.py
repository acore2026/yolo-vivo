"""Evaluate reviewed user-test samples; full-video inference is a prerequisite.

Annotations must be created by actual visual review before running this script.
This script never uses predictions to generate ground truth or select thresholds.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from audit_validation import match, panel
from common import ROOT, sha

NAMES = ['芍药', '原兰花', '原向日葵', '桉树叶']


def metrics(rows):
    stats = [dict(class_id=c, tp=0, fp=0, fn=0) for c in range(4)]
    for row in rows:
        for event in row['matching']['predictions']:
            c = row['predictions'][event['prediction']]['class']
            stats[c]['tp' if event['kind'] == 'tp' else 'fp'] += 1
        for index in row['matching']['missed_gt']:
            stats[row['gt'][index]['class']]['fn'] += 1
    for s in stats:
        s['precision'] = s['tp'] / (s['tp'] + s['fp']) if s['tp'] + s['fp'] else None
        s['recall'] = s['tp'] / (s['tp'] + s['fn']) if s['tp'] + s['fn'] else None
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    summary = json.loads((run / 'summary.json').read_text())
    model_label = summary.get('model_label', 'v3 cascade')
    ordinary = summary.get('model_kind') == 'ordinary_yolo'
    annotation_file = run / 'review/annotations.json'
    annotations = json.loads(annotation_file.read_text())
    photo = [json.loads(x) for x in (run / '01_test.jsonl').read_text().splitlines()]
    video = [json.loads(x) for x in (run / '02_test.jsonl').read_text().splitlines()]
    assert len(photo) == 1
    assert len(video) == summary['inputs'][1]['frames']
    assert [r['frame'] for r in video] == list(range(len(video)))
    pred = {r['frame']: r['detections'] for r in video}
    hashes = {r['source']: r['source_sha256'] for r in summary['inputs']}
    for source, expected in hashes.items():
        assert sha(ROOT / source) == expected
    for r in annotations:
        assert r['review'] == 'assistant_visual' and not r['human_review']
        assert r['source_sha256'] == hashes[r['source']]
        preview = run / 'review' / r['preview']
        assert sha(preview) == r['preview_sha256']
        r['predictions'] = photo[0]['detections'] if r['frame'] is None else pred[r['frame']]
        r['matching'] = match(r['gt'], r['predictions'])
        # Existing preview coordinates are 1280px wide; all GT/prediction boxes
        # are original source coordinates. Scale them together for the panels.
        im = cv2.imread(str(preview))
        scale = 1280 / (5712 if r['frame'] is None else 1920)
        scaled = lambda boxes: [dict(b, xyxy=[v * scale for v in b['xyxy']]) for b in boxes]
        comparison = np.hstack([
            panel(im, scaled(r['gt']), r['id'] + ' | assistant-reviewed GT'),
            panel(im, scaled(r['predictions']), r['id'] + ' | ' + model_label),
        ])
        assert cv2.imwrite(str(run / 'review' / f"compare_{r['id']}.jpg"), comparison)
    report = {
        'scope': 'Known user-test material regression; video metrics on 12 evenly spaced reviewed frames only',
        'reviewer': 'assistant_visual', 'human_review': False,
        'annotation_sha256': sha(annotation_file),
        'model_label': model_label,
        'model_sha256': summary.get('model_sha256', summary.get('classifier_sha256')),
        'annotation_reuse': (json.loads((run / 'review/annotation_reuse.json').read_text())
                             if (run / 'review/annotation_reuse.json').exists() else None),
        'operating_point': summary['operating_point'],
        'video_frames_inferred': len(video),
        'video_frames_reviewed': sum(r['frame'] is not None for r in annotations),
        'photo': metrics([r for r in annotations if r['frame'] is None]),
        'video_sample': metrics([r for r in annotations if r['frame'] is not None]),
        'match_iou': 0.5, 'rows': annotations,
    }
    target = run / 'review/metrics.json'
    with target.open('x') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    lines = ['# 用户测试素材复测', '',
             '模型：' + model_label + '；固定该模型训练流程的验证集阈值，未因本次结果调整参数。', '',
             f'完整推理：test.JPG 一张，test.MOV {len(video)} 帧。视频输出无音频。', '',
             '使用已实际查看并标注的照片及 12 个均匀视频抽样帧，共 48 个整盆目标；若复用旧标注，复用来源另存 annotation_reuse.json。审核为 assistant_visual，未经人工验收。', '',
             '以下视频指标只代表抽样帧，不代表全视频逐帧准确率。采用类别一致且 IoU≥0.5 的一对一匹配。', '']
    for key, title in [('photo', '照片'), ('video_sample', '视频抽样')]:
        lines += [f'## {title}', '', '|类别|TP|FP|FN|准确率 P|召回率 R|', '|---|---:|---:|---:|---:|---:|']
        for s in report[key]:
            fmt = lambda v: '未定义' if v is None else f'{100*v:.2f}%'
            lines.append(f"|{NAMES[s['class_id']]}|{s['tp']}|{s['fp']}|{s['fn']}|{fmt(s['precision'])}|{fmt(s['recall'])}|")
        lines.append('')
    lines += ['## 解释与限制', '',
              ('类别按用户原四类实物身份；普通 YOLO 无 CLIP 文本提示词。' if ordinary else '类别按用户原四类实物身份；lily / chrysanthemum 为现有模型提示词，非植物学鉴定。'),
              '这些素材之前已用于查看效果，本次属于回归检查，不能作为全新独立泛化验证。样本少且视频相邻帧相关，不能保证四类在实际场景 P/R 均超过 90%。',
              '未审核其余视频帧，不用检测次数或模型置信度代替准确率。没有训练、调阈值或部署。', '']
    (run / 'REPORT_CN.md').write_text('\n'.join(lines))
    html = '<meta charset="utf-8"><h1>用户测试复测</h1><p>左：assistant_visual 标注，右：' + model_label + ' 预测。原兰花/向日葵按原实物身份。</p>'
    html += ''.join(f'<h2>{r["id"]}</h2><img style="max-width:100%" src="compare_{r["id"]}.jpg">' for r in annotations)
    (run / 'review/index.html').write_text(html)
    print(json.dumps({k: report[k] for k in ['photo', 'video_sample', 'video_frames_inferred']}, indent=2))


if __name__ == '__main__':
    main()
