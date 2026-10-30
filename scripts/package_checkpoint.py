"""Package a fully trained/evaluated complete checkpoint; reject smoke-test weights."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from utils.checkpoint import load_model,validate_checkpoint
from utils.runtime import atomic_save,sha256,source_manifest,canonical_hash

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--source',required=True,type=Path)
    p.add_argument('--metrics',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--manifest',required=True,type=Path)
    args=p.parse_args();torch.set_num_threads(2)
    if args.output.exists():raise FileExistsError(args.output)
    model,state,identity=load_model(args.source,'cpu')
    if identity['run_kind']!='training' or model.variant!='segdis' or not state.get('epoch',0):
        raise ValueError('Only trained complete SegDis weights can become the public model release')
    sources=canonical_hash(source_manifest())
    if state.get('source_fingerprint')!=sources:raise ValueError('Packaging source differs from training source')
    done=json.loads((args.source.parent/'complete.json').read_text(encoding='utf-8'))
    if not done.get('complete') or done.get('best_checkpoint_sha256')!=identity['checkpoint_sha256']:
        raise ValueError('Training completion record does not identify this best checkpoint')
    metrics=json.loads(args.metrics.read_text(encoding='utf-8'))
    if metrics.get('images')!=3600 or metrics.get('identity')!=identity:
        raise ValueError('Complete matching USC12K evaluation is required')
    if metrics.get('protocol',{}).get('name')!='uscnet_raw_three_class_352':
        raise ValueError('Expected the fixed USC12K three-class protocol')
    if any(metrics['scenes'][f'Scene-{x}']['images']!=900 for x in 'ABCD'):
        raise ValueError('Expected 900 evaluation images per scene')
    keys=('format_version','architecture','model_variant','backbone_name','residual_input_mode',
          'model_schema','model_state','epoch','image_size','run_kind','training_config',
          'source_fingerprint','source_manifest','data_identity','backbone_sha256','runtime_versions')
    exported={k:state[k] for k in keys}
    exported['source_checkpoint_sha256']=identity['checkpoint_sha256']
    atomic_save(args.output,exported,tensor=True)
    loaded=torch.load(args.output,map_location='cpu',weights_only=True)
    validate_checkpoint(loaded,model)
    for key,value in state['model_state'].items():
        torch.testing.assert_close(value,loaded['model_state'][key],rtol=0,atol=0)
    atomic_save(args.manifest,{'status':'packaged_not_published','architecture':'segdis_rgb_v2',
        'model_variant':'segdis','asset':args.output.name,'sha256':sha256(args.output),
        'source_checkpoint_sha256':identity['checkpoint_sha256'],'source_fingerprint':sources,
        'parameter_count':sum(p.numel() for p in model.parameters()),'url':None})
    print(str(args.output.resolve()),flush=True)

if __name__=='__main__':main()
