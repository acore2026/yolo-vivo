"""Device-aware serving path; keep training/evaluation helpers unchanged.

Retains FP32, batch size 16, crop/preprocess, feature normalization, detector
thresholds and score calculation used by train.three_flower_cascade.
"""
import numpy as np
import torch


def features_on_device(model, preprocess, images):
    device = next(model.parameters()).device
    out = []
    with torch.inference_mode():
        for i in range(0, len(images), 16):
            batch = torch.stack([preprocess(image) for image in images[i:i + 16]])
            batch = batch.to(device=device, dtype=torch.float32)
            feature = model.encode_image(batch).float()
            out.append(feature / feature.norm(dim=-1, keepdim=True))
    return torch.cat(out) if out else torch.empty((0, 512), device=device, dtype=torch.float32)


def classify_boxes(encoder, preprocess, head, image, boxes):
    from flower_classifier import crop_image
    features = features_on_device(encoder, preprocess,
                                  [crop_image(image, box[:4]) for box in boxes])
    with torch.inference_mode():
        scores = head(features).softmax(1).cpu().tolist()
    result = []
    for box, score in zip(boxes, scores):
        cls = int(np.argmax(score))
        result.append({'class': cls, 'xyxy': box[:4],
                       'confidence': box[4] * score[cls],
                       'detector_confidence': box[4],
                       'classifier_confidence': score[cls]})
    return result


def candidates(detector, encoder, preprocess, head, image, device):
    boxes = detector.predict(np.asarray(image)[:, :, ::-1].copy(), device=device,
                             imgsz=640, conf=.01, iou=.7, verbose=False)[0].boxes.data.cpu().tolist()
    return classify_boxes(encoder, preprocess, head, image, boxes)
