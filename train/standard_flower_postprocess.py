"""Generic same-class nested-box suppression after ordinary YOLO NMS."""


def filter_predictions(predictions, conf, containment=None):
    thresholds = conf if isinstance(conf, (list, tuple)) else [conf] * 4
    if len(thresholds) != 4:
        raise ValueError('Expected four class thresholds')
    accepted = []
    for p in sorted(predictions, key=lambda x: -x['confidence']):
        if p['confidence'] < thresholds[p['class']]:
            continue
        suppress = False
        if containment is not None:
            if not 0 < containment <= 1:
                raise ValueError('Invalid containment fraction')
            for q in accepted:
                if p['class'] != q['class']:
                    continue
                a, b = p['xyxy'], q['xyxy']
                intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
                min_area = min((a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1]))
                if min_area > 0 and intersection / min_area >= containment:
                    suppress = True
                    break
        if not suppress:
            accepted.append(p)
    return accepted

