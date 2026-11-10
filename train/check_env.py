"""Read-only package and hardware report; no model imports/downloads."""
import importlib.metadata
import importlib.util
import json
import subprocess
from common import ROOT, sha

def main():
    report = {'packages': {}, 'base_model': {}}
    for name in ('torch','torchvision','ultralytics','numpy','PyYAML','Pillow'):
        try: report['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: report['packages'][name] = None
    report['clip_module_available'] = importlib.util.find_spec('clip') is not None
    model = ROOT/'models/yolov8s-worldv2.pt'
    report['base_model'] = {'path':str(model),'exists':model.exists(),'sha256':sha(model) if model.exists() else None}
    try:
        p = subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.free','--format=csv,noheader'],capture_output=True,text=True,timeout=10)
        report['gpu'] = {'returncode':p.returncode,'output':p.stdout+p.stderr}
    except (OSError, subprocess.TimeoutExpired) as e: report['gpu'] = {'error':str(e)}
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__ == '__main__': main()
