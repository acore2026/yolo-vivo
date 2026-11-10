"""Check this environment's CUDA execution; no training or dataset mutation."""
import json
import sys
import torch

def main():
    report = {'python':sys.executable, 'torch':torch.__version__,
              'cuda_build':torch.version.cuda, 'cuda_available':torch.cuda.is_available()}
    if not report['cuda_available']:
        print(json.dumps(report,indent=2))
        raise SystemExit('CUDA unavailable in this process; run in the ordinary WSL terminal.')
    try:
        x = torch.ones((32,32),device='cuda:0')
        value = (x @ x).mean().item()
        torch.cuda.synchronize()
        if value != 32.0:
            raise RuntimeError('CUDA calculation produced an unexpected result')
        report.update(device=torch.cuda.get_device_name(0),
                      memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                      cuda_matmul='PASS')
    except Exception as exc:
        report['error'] = str(exc)
        print(json.dumps(report,indent=2))
        raise SystemExit(1)
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    main()
