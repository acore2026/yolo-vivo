"""CPU numerical checks. No dataset, optimization, CLIP, downloads or training."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import io
import json
import os
import tempfile
import torch
from common import resolve, sha
from lora import LoRAConfig, LoRALinear, LoRAConv2d, apply_lora_adapters, configure_lora_trainable, merge_and_strip_lora

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default='models/yolov8s-worldv2.pt')
    p.add_argument('--output',default='train/LORA_CHECK_REPORT.json')
    a=p.parse_args();torch.manual_seed(42);torch.set_num_threads(2)
    checks={}
    for name,base,x in [('linear',torch.nn.Linear(12,8),torch.randn(2,12)),('conv',torch.nn.Conv2d(6,8,3,padding=1),torch.randn(2,6,16,16))]:
        layer=(LoRALinear if name=='linear' else LoRAConv2d)(deepcopy(base),4,8,0).eval()
        with torch.no_grad():
            torch.testing.assert_close(base(x),layer(x),atol=0,rtol=0)
            layer.lora_B.normal_(std=0.01)
            before=layer(x);merged=layer.merged_module()(x)
            torch.testing.assert_close(before,merged,atol=2e-6,rtol=2e-5)
        buf=io.BytesIO();torch.save(layer.state_dict(),buf);buf.seek(0)
        restored=deepcopy(layer);restored.load_state_dict(torch.load(buf,weights_only=True))
        torch.testing.assert_close(restored(x),before,atol=0,rtol=0)
        checks[name]={'zero_init':'PASS','merge_max_abs':float((before-merged).abs().max()),'save_reload':'PASS'}
    # Isolate framework config writes; no text encoding or set_classes calls.
    with tempfile.TemporaryDirectory(prefix='flower-lora-check-') as temp:
        os.environ['YOLO_CONFIG_DIR']=temp
        os.environ['YOLO_AUTOINSTALL']='false'
        from ultralytics import YOLOWorld
        base_path=resolve(a.model);digest=sha(base_path)
        model=YOLOWorld(str(base_path)).model.float().eval()
        baseline=deepcopy(model)
        matched=apply_lora_adapters(model,LoRAConfig())
        count,total=configure_lora_trainable(model,'none')
        names=[n for n,p in model.named_parameters() if p.requires_grad]
        assert names and all(n.endswith(('lora_A','lora_B')) for n in names)
        x=torch.randn(1,3,64,64)
        with torch.no_grad():
            expected=baseline(x)[0];actual=model(x)[0]
            torch.testing.assert_close(actual,expected,atol=0,rtol=0)
            for n,v in model.named_parameters():
                if n.endswith('lora_B'): v.normal_(std=0.001)
            adapted=model(x)[0]
            merged=merge_and_strip_lora(deepcopy(model)).eval()(x)[0]
            torch.testing.assert_close(adapted,merged,atol=0.001,rtol=0.0002)
        # Exercise the real adapter artifact loader, not just layer state_dict.
        from pathlib import Path
        from model_io import load_model
        artifact=Path(temp)/'synthetic_adapter.pt'
        torch.save({'format_version':1,'base_sha256':digest,'config':asdict(LoRAConfig()),
                    'matched_modules':matched,'state_dict':{n:v.cpu() for n,v in model.state_dict().items() if n.endswith(('lora_A','lora_B'))}},artifact)
        loaded=load_model(base_path,artifact).model.float().eval()
        with torch.no_grad():
            restored=loaded(x)[0]
            torch.testing.assert_close(restored,merged,atol=0,rtol=0)
        assert digest==sha(base_path)
        checks['actual_model']={'base_sha256':digest,'matched_modules':matched,'trainable_parameters':count,'total_parameters':total,'zero_init_exact':'PASS','merge_max_abs':float((adapted-merged).abs().max()),'adapter_artifact_reload_exact':'PASS','base_file_unchanged':True}
    report={'result':'PASS','device':'cpu','scope':'synthetic tensors only; no optimization or dataset training','checks':checks,'not_tested':['full trainer loop','optimizer step','CLIP prompts','GPU','resume','accuracy']}
    resolve(a.output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
