"""Project-local framework settings and verified offline CLIP cache."""
import os
from common import resolve, sha

CLIP_SHA256 = '40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'

def setup():
    os.environ['YOLO_CONFIG_DIR'] = str(resolve('train/.cache/ultralytics'))
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    weights = resolve('train/.cache/weights')
    clip = weights/'clip/ViT-B-32.pt'
    if not clip.is_file() or sha(clip) != CLIP_SHA256:
        raise RuntimeError('Verified local CLIP weights required; automatic download disabled by preflight')
    from ultralytics import settings
    import ultralytics.utils as utils
    settings.update({'weights_dir':str(weights), 'datasets_dir':str(resolve('train/datasets')),
                     'runs_dir':str(resolve('train/runs')), 'sync':False})
    # WEIGHTS_DIR is captured at import time by the installed version.
    utils.WEIGHTS_DIR = weights
    import ultralytics.nn.text_model as text_model
    text_model.WEIGHTS_DIR = weights
