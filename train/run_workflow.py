"""Approved GPU workflow: 1-epoch smoke -> fresh full training -> held-out test."""
import argparse
from copy import deepcopy
from datetime import datetime
import json
import subprocess
import sys
import traceback
import yaml
from common import ROOT, config, resolve, sha

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='train/configs/flowers.yaml')
    parser.add_argument('--check-only',action='store_true',help='No training, no workflow directory')
    parser.add_argument('--eval-split', choices=['val','test'], default='test')
    a=parser.parse_args();c=config(a.config)
    # Fail before creating a run if the execution environment cannot access CUDA.
    subprocess.run([sys.executable,str(ROOT/'train/check_gpu.py')],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(ROOT/'train/train.py'),'--config',str(resolve(a.config))],cwd=ROOT,check=True)
    if a.check_only: return
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    folder=resolve('train/workflows')/stamp;folder.mkdir(parents=True,exist_ok=False)
    original_hash=sha(resolve(c['model']))
    configs={}
    for stage,epochs in [('smoke',1),('full',c['training']['epochs'])]:
        cfg=deepcopy(c);cfg['name']=f'flowers_{stamp}_{stage}'
        cfg['training']['epochs']=epochs
        if stage=='smoke': cfg['training']['patience']=0
        path=folder/f'{stage}.yaml';path.write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False));configs[stage]=(cfg,path)
    state={'status':'running','folder':str(folder),'base_sha256':original_hash,'completed':[]}
    def write(): (folder/'status.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')
    def run(stage,script,args):
        state['current_stage']=stage;write()
        print(f'\n[{stage}] log: {folder/stage}.log',flush=True)
        command=[sys.executable,'-u',str(ROOT/'train'/script),*args]
        with (folder/f'{stage}.log').open('w') as log:
            with subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1) as child:
                try:
                    for line in child.stdout:
                        print(line,end='',flush=True);log.write(line);log.flush()
                    code=child.wait()
                    if code: raise subprocess.CalledProcessError(code,command)
                except BaseException:
                    child.terminate()
                    try: child.wait(timeout=15)
                    except subprocess.TimeoutExpired: child.kill();child.wait()
                    raise
        state['completed'].append(stage);write()
    write()
    try:
        for stage in ('smoke','full'):
            cfg,path=configs[stage]
            run(stage,'train.py',['--config',str(path),'--run'])
            run(stage+'_verify','verify_run.py',['--config',str(path)])
            if sha(resolve(c['model']))!=original_hash: raise RuntimeError('Base weights changed')
        full,_=configs['full'];adapter=resolve(full['project'])/full['name']/'weights/adapter_best.pt'
        run('final_'+a.eval_split,'evaluate.py',['--base',str(resolve(c['model'])),'--adapter',str(adapter),
            '--data',str(resolve(c['data'])),'--split',a.eval_split,'--device',str(c['training']['device']),
            '--output',str(folder/(a.eval_split+'_evaluation'))])
        state['status']='complete';state['best_adapter']=str(adapter);write()
        print(f'Workflow complete: {folder}',flush=True)
    except BaseException as exc:
        state['status']='failed';state['error']=str(exc);write()
        (folder/'error.txt').write_text(traceback.format_exc())
        raise

if __name__=='__main__': main()
