"""Verify the current public source/model; optionally test real 352px images."""
import argparse
import ast
from pathlib import Path
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from models import build_model
from utils.checkpoint import model_metadata,load_model
from utils.runtime import ROOT,atomic_save,source_manifest,canonical_hash

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--usc_root',type=Path)
    p.add_argument('--backbone_checkpoint',type=Path)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args();torch.set_num_threads(2);torch.manual_seed(42)
    for source in (ROOT/'models').glob('*.py'):
        tree=ast.parse(source.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node,(ast.FunctionDef,ast.ClassDef,ast.Attribute)):
                name=getattr(node,'name',getattr(node,'attr','')).lower()
                if any(x in name for x in ('frequency','haar','wavelet','fourier')):
                    raise AssertionError(f'Unsupported model implementation: {source.name}:{name}')
    model=build_model(str(args.backbone_checkpoint) if args.backbone_checkpoint else '').eval()
    channels=[b.fusion[0][0].in_channels for b in model.foreground.scale_blocks]+[
        model.foreground.boundary_block.smooth_body[0][0].in_channels,model.foreground.boundary_block.sharp_body[0][0].in_channels]
    assert channels==[290,290,290,291,291]
    assert sum(p.numel() for p in model.parameters())==28376678
    assert not any('frequency' in k.lower() or 'haar' in k.lower() for k in model.state_dict())
    cases=[]
    if args.usc_root:
        from dataset_usc12k import USC12KDataset
        from utils.data_audit import audit_usc12k
        data_audit=audit_usc12k(args.usc_root)
        for scene in ('Scene-A','Scene-B','Scene-C','Scene-D'):
            sample=USC12KDataset(args.usc_root,scene,352,False)[0]
            with torch.no_grad():out=model(sample['image'][None])
            assert out['three_class_probability'].shape==(1,3,352,352)
            assert torch.isfinite(out['three_class_probability']).all()
            assert torch.equal(out['residual_input'][:,:3],sample['image'][None])
            cases.append({'scene':scene,'sample_id':sample['sample_id'],'shape':[1,3,352,352]})
        image=sample['image'][None]
    else:
        image=torch.randn(1,3,64,64)
        data_audit=None
    with torch.no_grad():reference=model(image)['three_class_probability']
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'smoke.pth'
        torch.save({**model_metadata(model),'model_state':model.state_dict(),'epoch':0,
                    'image_size':352,'run_kind':'smoke_test'},path)
        restored,state,identity=load_model(path,'cpu')
        with torch.no_grad():actual=restored(image)['three_class_probability']
        torch.testing.assert_close(reference,actual,rtol=0,atol=0)
    sources=source_manifest()
    report={'passed':True,'architecture':model.architecture,'parameter_count':28376678,
            'fusion_in_channels':channels,'strict_checkpoint_roundtrip':True,'real_image_cases':cases,
            'source_fingerprint':canonical_hash(sources),'source_manifest':sources,
            'data_audit':data_audit,'benchmark_accuracy_verified':False,'trained_checkpoint_available':False}
    atomic_save(args.output,report)
    print(str(args.output.resolve()),flush=True)

if __name__=='__main__':main()
