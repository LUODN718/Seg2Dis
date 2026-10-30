"""Strict model identity checks for every training/inference entry point."""
from pathlib import Path
import torch
from config import CFG
from .runtime import canonical_hash, sha256

FORMAT_VERSION=1

def model_schema(state):
    return canonical_hash({k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in sorted(state.items())})

def model_metadata(model):
    return {'format_version':FORMAT_VERSION,'architecture':CFG.architecture,
            'model_variant':model.variant,'backbone_name':model.backbone_name,
            'residual_input_mode':model.foreground.residual_input_mode,
            'model_schema':model_schema(model.state_dict())}

def validate_checkpoint(checkpoint, model=None):
    from models.factory import VARIANTS
    if checkpoint.get('format_version')!=FORMAT_VERSION or checkpoint.get('architecture')!=CFG.architecture:
        raise ValueError('Incompatible checkpoint format/architecture; no automatic weight conversion is supported')
    if checkpoint.get('model_variant') not in VARIANTS or checkpoint.get('backbone_name')!='pvt_v2_b2':
        raise ValueError('Unknown checkpoint model variant or backbone')
    if checkpoint.get('residual_input_mode') not in ('rgb','rgb_pm_u'):
        raise ValueError('Unknown residual input mode')
    state=checkpoint.get('model_state')
    if not isinstance(state,dict) or not state:
        raise ValueError('Checkpoint model_state is missing or empty')
    if any('frequency' in k.lower() or 'haar' in k.lower() for k in state):
        raise ValueError('Checkpoint contains unsupported model branches')
    if checkpoint.get('model_schema')!=model_schema(state):
        raise ValueError('Checkpoint tensor schema does not match its metadata')
    if model is not None:
        if (model.variant!=checkpoint['model_variant'] or
                model.foreground.residual_input_mode!=checkpoint['residual_input_mode'] or
                model_schema(model.state_dict())!=checkpoint['model_schema']):
            raise ValueError('Checkpoint does not match the requested model variant/tensor shapes')

def load_model(path, device='cpu'):
    from models import build_model
    state=torch.load(path,map_location='cpu',weights_only=False)
    validate_checkpoint(state)
    model=build_model(backbone_checkpoint='',backbone_name=state['backbone_name'],
                      variant=state['model_variant'],
                      residual_input_mode=state['residual_input_mode'])
    validate_checkpoint(state,model)
    model.load_state_dict(state['model_state'],strict=True)
    model.to(device).eval()
    identity={k:state[k] for k in ('architecture','model_variant','backbone_name','model_schema')}
    identity.update(checkpoint_sha256=sha256(path),selected_epoch=state.get('epoch'),
                    experiment_id=state.get('training_config',{}).get('id'),
                    source_fingerprint=state.get('source_fingerprint'),
                    run_kind=state.get('run_kind','unknown'))
    return model,state,identity
