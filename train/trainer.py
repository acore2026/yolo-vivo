"""Version-pinned LoRA trainer with separate adapter and raw resume snapshots."""
from copy import deepcopy
from dataclasses import asdict
import json
import random
import numpy as np
import torch
from ultralytics.models.yolo.world.train import WorldTrainer
from ultralytics.utils.torch_utils import unwrap_model
from lora import apply_lora_adapters, configure_lora_trainable, merge_and_strip_lora

class FlowerLoRATrainer(WorldTrainer):
    adapter_config = None
    base_sha256 = None
    camera_augmentation = False

    def build_dataset(self, img_path, mode='train', batch=None):
        if mode != 'train' or not self.camera_augmentation:
            return super().build_dataset(img_path, mode, batch)
        from flower_augmentation import build_flower_dataset
        stride = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
        dataset = build_flower_dataset(self.args, img_path, batch, self.data, stride)
        self.set_text_embeddings([dataset], batch)
        return dataset

    def build_optimizer(self, *args, **kwargs):
        optimizer = super().build_optimizer(*args, **kwargs)
        # Keep framework group order/warmup metadata, but exclude frozen tensors.
        for group in optimizer.param_groups:
            group['params'] = [p for p in group['params'] if p.requires_grad]
        actual = {id(p) for g in optimizer.param_groups for p in g['params']}
        expected = {id(p) for p in self.model.parameters() if p.requires_grad}
        if not actual or actual != expected:
            raise RuntimeError('Optimizer/trainable parameter mismatch')
        return optimizer

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg, weights, verbose)
        self.matched_modules = apply_lora_adapters(model, self.adapter_config)
        return model

    def _build_train_pipeline(self):
        configure_lora_trainable(self.model, 'none')
        super()._build_train_pipeline()
        names=[n for n,p in self.model.named_parameters() if p.requires_grad]
        if not names or any(not n.endswith(('lora_A','lora_B')) for n in names):
            raise RuntimeError('Non-LoRA parameters are trainable')
        if not hasattr(self, 'frozen_snapshot'):
            self.frozen_snapshot={n:p.detach().cpu().clone() for n,p in self.model.named_parameters() if not p.requires_grad}
            self.bn_snapshot={n:b.detach().cpu().clone() for n,b in self.model.named_buffers() if n.endswith(('running_mean','running_var','num_batches_tracked'))}
        (self.save_dir/'lora_verification.json').write_text(json.dumps({'trainable_names':names,'matched_modules':self.matched_modules,'base_sha256':self.base_sha256},indent=2))

    def _model_train(self):
        super()._model_train()
        for m in self.model.modules():
            if isinstance(m,torch.nn.modules.batchnorm._BatchNorm): m.eval()

    def save_model(self):
        model=unwrap_model(self.model)
        for name,p in model.named_parameters():
            if name in self.frozen_snapshot and not torch.equal(p.detach().cpu(),self.frozen_snapshot[name]):
                raise RuntimeError(f'Frozen parameter changed: {name}')
        for name,b in model.named_buffers():
            if name in self.bn_snapshot and not torch.equal(b.detach().cpu(),self.bn_snapshot[name]):
                raise RuntimeError(f'BatchNorm buffer changed: {name}')
        lora_tensors = [p for n,p in model.named_parameters() if n.endswith(('lora_A','lora_B'))]
        if not all(torch.isfinite(p).all().item() for p in lora_tensors):
            raise RuntimeError('Non-finite LoRA parameters')
        changed = any(torch.count_nonzero(p).item() > 0 for n,p in model.named_parameters() if n.endswith('lora_B'))
        if not changed:
            raise RuntimeError('LoRA B tensors remain zero; no learning detected')
        (self.save_dir/'training_integrity.json').write_text(json.dumps({
            'epoch':self.epoch,'frozen_parameters_unchanged':True,'batchnorm_unchanged':True,
            'lora_finite':True,'lora_updated':True,
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(self.device) if self.device.type=='cuda' else None,
            'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(self.device) if self.device.type=='cuda' else None,
        },indent=2))
        ema=self.ema.ema
        payload={'format_version':2,'base_sha256':self.base_sha256,'config':asdict(self.adapter_config),
                 'architecture':{'kind':'world_yaml_rebuild','yaml':deepcopy(model.yaml),'channels':self.data['channels'],'nc':min(self.data['nc'],80)},
                 'matched_modules':self.matched_modules,'names':self.data['names'],
                 'state_dict':{n:p.detach().cpu() for n,p in unwrap_model(ema).state_dict().items() if n.endswith(('lora_A','lora_B'))}}
        self.wdir.mkdir(parents=True,exist_ok=True)
        torch.save(payload,self.wdir/'adapter_last.pt')
        if self.fitness==self.best_fitness: torch.save(payload,self.wdir/'adapter_best.pt')
        # Preserve full unmerged state. Resume restoration is intentionally not enabled until runtime-tested.
        torch.save({'model':model.state_dict(),'ema':ema.state_dict(),'optimizer':self.optimizer.state_dict(),
                    'scheduler':self.scheduler.state_dict(),'scaler':self.scaler.state_dict(),'epoch':self.epoch,
                    'ema_updates':self.ema.updates,'best_fitness':self.best_fitness,
                    'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                    'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
                    'adapter_metadata':{k:v for k,v in payload.items() if k!='state_dict'}},self.wdir/'training_state_last.pt')
        try:
            self.ema.ema=merge_and_strip_lora(deepcopy(unwrap_model(ema))).half()
            super().save_model()
        finally: self.ema.ema=ema
