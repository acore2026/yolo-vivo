"""Adapter loading for plain YOLO-World. Base hash and keys are strict."""
from common import sha

def load_model(base,adapter=None):
    import torch
    from runtime import setup
    setup()
    from ultralytics import YOLOWorld
    model=YOLOWorld(str(base))
    if adapter:
        from copy import deepcopy
        from lora import LoRAConfig,apply_lora_adapters,merge_and_strip_lora
        payload=torch.load(adapter,map_location='cpu',weights_only=True)
        if payload['base_sha256']!=sha(base): raise ValueError('Adapter/base SHA256 mismatch')
        if payload.get('format_version',1)==2:
            from ultralytics.nn.tasks import WorldModel
            architecture=payload['architecture']
            if architecture['kind']!='world_yaml_rebuild': raise ValueError('Unsupported adapter architecture')
            rebuilt=WorldModel(deepcopy(architecture['yaml']),ch=architecture['channels'],nc=architecture['nc'],verbose=False)
            rebuilt.load(model.model,verbose=False)
            model.model=rebuilt
        elif payload.get('format_version',1)!=1:
            raise ValueError('Unsupported adapter format version')
        matched=apply_lora_adapters(model.model,LoRAConfig(**payload['config']))
        if matched!=payload['matched_modules']: raise ValueError('Adapter target modules differ')
        expected={n for n in model.model.state_dict() if n.endswith(('lora_A','lora_B'))}
        if expected!=set(payload['state_dict']): raise ValueError('Adapter tensor keys differ')
        model.model.load_state_dict(payload['state_dict'],strict=False)
        model.model=merge_and_strip_lora(model.model).eval()
    return model
