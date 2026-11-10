"""Small residual adapter on frozen CLIP image features; original encoder unchanged."""
import torch
from runtime import CLIP_SHA256

class VisualFeatureAdapter(torch.nn.Module):
    def __init__(self, classes=5, bottleneck=64):
        super().__init__()
        self.down = torch.nn.Linear(512, bottleneck)
        self.up = torch.nn.Linear(bottleneck, 512)
        self.classifier = torch.nn.Linear(512, classes)
        torch.nn.init.zeros_(self.up.weight)
        torch.nn.init.zeros_(self.up.bias)

    def forward(self, x):
        adapted = x + self.up(torch.nn.functional.gelu(self.down(x)))
        return self.classifier(torch.nn.functional.normalize(adapted, dim=-1))


def load_classifier(payload):
    if payload['encoder_sha256'] != CLIP_SHA256:
        raise ValueError('Classifier/CLIP encoder mismatch')
    kind = payload.get('head_kind', 'linear')
    if kind == 'linear':
        model = torch.nn.Linear(512, payload['head']['weight'].shape[0])
    elif kind == 'visual_feature_adapter':
        model = VisualFeatureAdapter(payload['num_classes'], payload['bottleneck'])
    else:
        raise ValueError(f'Unsupported classifier: {kind}')
    model.load_state_dict(payload['head'], strict=True)
    return model.eval()
