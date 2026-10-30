"""One trainer for current SegDis and its explicitly configured controls."""
import argparse
import json
import os
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch
from torch.nn import functional as F
from models import build_model, validate_checkpoint
from utils.checkpoint import model_metadata
from utils.experiment import load_config, make_dataset, make_loader, data_identity
from utils.loss import structure_loss
from utils.runtime import (ROOT, atomic_save, canonical_hash, resolve_device,
                           seed_everything, sha256, source_manifest, runtime_versions)

def compute_objective(output,batch,variant,task,device,type_weight=0.5,calibration_weight=0.5):
    target=batch['semantic'].to(device,non_blocking=True)
    foreground=(target>0).float()[:,None]
    if task=='JOINT':
        ids=batch['task_id'].to(device)
        if not all(bool((ids==k).any()) for k in (0,1)):
            raise ValueError('Every JOINT batch must contain intact SOD/COD pairs')
        fg=sum(structure_loss(output['union'][ids==k],foreground[ids==k]) for k in (0,1))/2
    else:
        fg=structure_loss(output['union'],foreground)
    zero=fg.new_zeros(())
    terms={'fg':fg,'ce':zero,'type':zero,'composed':zero}
    if variant=='flr': loss=fg
    elif variant in ('segdis','segdis_no_correction','segdis_no_rta'):
        valid=target>0
        if bool(valid.any()):
            terms['type']=F.cross_entropy(output['semantic_logits'].permute(0,2,3,1)[valid],(target[valid]==2).long())
        terms['composed']=F.nll_loss(output['three_class_probability'].clamp_min(1e-6).log(),target)
        loss=fg+type_weight*terms['type']+calibration_weight*terms['composed']
    else:
        terms['ce']=F.cross_entropy(output['logits'],target)
        loss=fg+terms['ce']
    return loss,terms

def compute_loss(model,batch,device,type_weight=0.5,calibration_weight=0.5,task='USC12K'):
    output=model(batch['image'].to(device,non_blocking=True))
    loss,terms=compute_objective(output,batch,getattr(model,'variant','segdis'),task,device,type_weight,calibration_weight)
    return loss,(terms['fg'],terms['type'],terms['composed'])

def make_optimizer(model,config):
    backbone=list(model.v3.backbone.parameters())
    ids={id(p) for p in backbone}
    others=[p for p in model.parameters() if id(p) not in ids]
    groups=[{'params':backbone,'lr':config['backbone_lr']},{'params':others,'lr':config['base_lr']}]
    assigned=[id(p) for g in groups for p in g['params']]
    if len(assigned)!=len(set(assigned)) or set(assigned)!={id(p) for p in model.parameters()}:
        raise RuntimeError('Optimizer does not cover every model parameter exactly once')
    return torch.optim.AdamW(groups,weight_decay=config['weight_decay'])

def parse_args():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--usc_root',type=Path)
    parser.add_argument('--sod_train',type=Path)
    parser.add_argument('--cod_train',type=Path)
    parser.add_argument('--backbone_checkpoint',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--smoke_test',action='store_true',help='One real-data update only; not releasable training')
    return parser.parse_args()

def main():
    args=parse_args()
    if args.preflight:
        os.environ['CUDA_VISIBLE_DEVICES']='-1'
        device=torch.device('cpu')
    else: device=resolve_device(args.device)
    if device.type=='cpu': torch.set_num_threads(2)
    config=load_config(args.config)
    seed_everything(config['seed'])
    generator=torch.Generator().manual_seed(config['seed'])
    dataset=make_dataset(config,args)
    loader=make_loader(config,dataset,generator,workers=0 if args.preflight else None)
    data=data_identity(dataset,args,config)
    sources=source_manifest()
    provenance={'training_config':config,'data_identity':data,'backbone_sha256':sha256(args.backbone_checkpoint),
                'source_manifest':sources,'source_fingerprint':canonical_hash(sources),
                'runtime_versions':runtime_versions()}
    model=build_model(str(args.backbone_checkpoint) if not args.resume else '',variant=config['model_variant']).to(device)
    optimizer=make_optimizer(model,config)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config['epochs'])
    metadata=model_metadata(model)
    args.output.mkdir(parents=True,exist_ok=True)
    if args.preflight:
        model.train()
        batch=next(iter(loader))
        # Validate the real loader, then use a small CPU backward test.
        subset=2 if config['task']=='JOINT' else 1
        batch={k:v[:subset] if torch.is_tensor(v) else v[:subset] for k,v in batch.items()}
        batch['image']=F.interpolate(batch['image'],(64,64),mode='bilinear',align_corners=False)
        batch['semantic']=F.interpolate(batch['semantic'][:,None].float(),(64,64),mode='nearest')[:,0].long()
        loss,_=compute_loss(model,batch,device,config['type_weight'],config['calibration_weight'],config['task'])
        loss.backward()
        if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise FloatingPointError('Preflight loss/backward is not finite')
        result={'passed':True,'device':'CPU','backward_size':64,'actual_loader_image_size':config['image_size'],
                'parameter_count':sum(p.numel() for p in model.parameters()),'train_images':len(dataset),
                'steps_per_epoch':len(loader),'metadata':metadata,'provenance':provenance}
        atomic_save(args.output/'preflight.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='provenance'},indent=2),flush=True)
        return
    config_path=args.output/'training_config.json'
    if config_path.exists():
        if not args.resume or json.loads(config_path.read_text(encoding='utf-8'))!=provenance:
            raise ValueError('Output directory already belongs to another run; explicit matching resume is required')
    atomic_save(config_path,provenance)
    for relative in sources:
        target=args.output/'source_snapshot'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and sha256(target)!=sources[relative]:
            raise ValueError(f'Frozen output source differs: {relative}')
        if not target.exists(): shutil.copy2(ROOT/relative,target)
    start,best,updates=0,float('inf'),0
    if args.resume:
        state=torch.load(args.resume,map_location='cpu',weights_only=False)
        validate_checkpoint(state,model)
        if state.get('run_kind')!='training' or any(state.get(k)!=v for k,v in provenance.items()):
            raise ValueError('Resume source/configuration/data/initialization differs')
        model.load_state_dict(state['model_state'],strict=True)
        optimizer.load_state_dict(state['optimizer_state']);scheduler.load_state_dict(state['scheduler_state'])
        start,best,updates=state['epoch'],state['best_loss'],state['training_updates']
        rng=state['rng'];random.setstate(rng['python']);np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['torch']);generator.set_state(rng['loader'])
        if device.type=='cuda': torch.cuda.set_rng_state_all(rng['cuda'])
        del state
    for epoch in range(start,config['epochs']):
        model.train();total=0.0;steps=0;tick=time.monotonic()
        for batch in loader:
            if args.smoke_test:
                n=2 if config['task']=='JOINT' else 1
                batch={k:v[:n] for k,v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss,_=compute_loss(model,batch,device,config['type_weight'],config['calibration_weight'],config['task'])
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training objective')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip'],error_if_nonfinite=True)
            optimizer.step();total+=float(loss.detach());steps+=1;updates+=1
            if args.smoke_test: break
        if not steps: raise RuntimeError('Empty training loader')
        if not args.smoke_test and steps!=config['expected_steps_per_epoch']:
            raise RuntimeError('Epoch did not complete the configured update budget')
        scheduler.step();average=total/steps
        improved=average<best;best=min(best,average)
        state={**metadata,**provenance,'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),
               'scheduler_state':scheduler.state_dict(),'epoch':0 if args.smoke_test else epoch+1,
               'training_updates':updates,'best_loss':best,'image_size':config['image_size'],
               'run_kind':'smoke_test' if args.smoke_test else 'training',
               'rng':{'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
                      'loader':generator.get_state(),'cuda':torch.cuda.get_rng_state_all() if device.type=='cuda' else []}}
        atomic_save(args.output/'latest.pth',state,tensor=True)
        if improved: atomic_save(args.output/'best.pth',state,tensor=True)
        row={'epoch':state['epoch'],'epochs':config['epochs'],'loss':average,'steps':steps,
             'seconds':time.monotonic()-tick,'run_kind':state['run_kind']}
        atomic_save(args.output/'status.json',row)
        with (args.output/'history.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        if args.smoke_test: return
    atomic_save(args.output/'complete.json',{'complete':True,'experiment_id':config['id'],
                'epochs':config['epochs'],'best_checkpoint_sha256':sha256(args.output/'best.pth'),
                'source_fingerprint':provenance['source_fingerprint'],'metadata':metadata})

if __name__=='__main__': main()
