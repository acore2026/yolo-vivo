"""Three-class GPU workflow: fresh LoRA, classifier, frozen tests; isolated outputs."""
import argparse,json,subprocess,sys
from datetime import datetime
from pathlib import Path
import yaml
from common import ROOT,sha
from three_flower_data import TARGET,verify,dump

def main():
 p=argparse.ArgumentParser();p.add_argument('--check-only',action='store_true');a=p.parse_args();rows,report=verify(TARGET);print(json.dumps(report,indent=2),flush=True)
 config_path=ROOT/'train/configs/flowers_three_v1.yaml';c=yaml.safe_load(config_path.read_text())
 subprocess.run([sys.executable,str(ROOT/'train/train.py'),'--config',str(config_path)],cwd=ROOT,check=True)
 if a.check_only:return
 subprocess.run([sys.executable,str(ROOT/'train/check_gpu.py')],cwd=ROOT,check=True)
 folder=ROOT/'train/workflows'/('three_flowers_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));folder.mkdir(exist_ok=False);state=dict(status='running',completed=[],base_sha256=sha(ROOT/c['model']),configuration=c)
 def run(stage,script,args):
  state['stage']=stage;dump(folder/'status.json',state)
  with (folder/f'{stage}.log').open('x') as log:
   proc=subprocess.Popen([sys.executable,'-u',str(ROOT/'train'/script),*args],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
   try:
    for line in proc.stdout:print(line,end='',flush=True);log.write(line);log.flush()
    if proc.wait():raise RuntimeError(stage+' failed; see '+str(folder/f'{stage}.log'))
   except BaseException:
    proc.terminate()
    try:proc.wait(timeout=15)
    except subprocess.TimeoutExpired:proc.kill();proc.wait()
    raise
  state['completed'].append(stage);dump(folder/'status.json',state)
 try:
  for stage,epochs in [('smoke',1),('full',c['training']['epochs'])]:
   cfg=json.loads(json.dumps(c));cfg['name']=folder.name+'_'+stage;cfg['training']['epochs']=epochs
   if stage=='smoke':cfg['training']['patience']=0
   path=folder/f'{stage}.yaml';path.write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False));run(stage,'train.py',['--config',str(path),'--run']);run(stage+'_verify','verify_run.py',['--config',str(path)])
  adapter=ROOT/cfg['project']/cfg['name']/'weights/adapter_best.pt';classifier=folder/'classifier'
  run('classifier','three_flower_cascade.py',['train','--adapter',str(adapter),'--output',str(classifier),'--device','0'])
  run('test','three_flower_cascade.py',['test','--classifier-dir',str(classifier),'--output',str(folder/'test'),'--test-data','train/datasets/flowers_three_test_v1','--device','0'])
  run('paired_evaluation','evaluate_three_user_samples.py',['--run',str(folder/'test')])
  run('introductions','../scripts/render_flower_introductions.py',['--three-classes','--source','test_videos/test2.MOV','--predictions-dir',str(folder/'test'),'--output',str(folder/'introductions')])
  for video in [folder/'test/02_test.mp4',folder/'test/03_test2.mp4',folder/'introductions/preview.mp4']:
   subprocess.run(['ffmpeg','-nostdin','-v','error','-n','-i',str(video),'-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(video.with_name(video.stem+'_h264.mp4'))],check=True)
  assert sha(ROOT/c['model'])==state['base_sha256'];verify(TARGET);state.update(status='completed',best_adapter=str(adapter),classifier=str(classifier),test_output=str(folder/'test'));dump(folder/'status.json',state)
 except BaseException as exc:
  state.update(status='failed',error=str(exc));dump(folder/'status.json',state);raise
 print('Completed:',folder,flush=True)
if __name__=='__main__':main()
