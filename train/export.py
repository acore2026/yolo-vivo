"""Merge an adapter into a NEW inference model, never overwrite the base."""
import argparse
from common import resolve,sha

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',default='models/yolov8s-worldv2.pt')
    p.add_argument('--adapter',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args();base=resolve(a.base);out=resolve(a.output)
    if out.exists(): raise SystemExit('Output exists; refusing overwrite')
    from model_io import load_model
    original=sha(base);model=load_model(base,resolve(a.adapter))
    out.parent.mkdir(parents=True,exist_ok=True);model.save(str(out))
    if sha(base)!=original: raise RuntimeError('Base changed')

if __name__=='__main__': main()
