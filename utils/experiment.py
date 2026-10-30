"""Validated experiment definitions and data construction."""
import json
from pathlib import Path
from torch.utils.data import DataLoader
from models.factory import VARIANTS
from dataset_usc12k import USC12KDataset
from datasets import BinaryTaskDataset, JointTaskDataset, PairedBatchSampler
from .runtime import canonical_hash, seed_worker, sha256

FIELDS={'id','model_variant','task','epochs','image_size','batch_size','workers','seed',
        'base_lr','backbone_lr','weight_decay','gradient_clip','type_weight','calibration_weight',
        'sampling','selection','expected_train_images','expected_steps_per_epoch'}

def load_config(path):
    config=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if set(config)!=FIELDS: raise ValueError(f'Configuration fields differ: {set(config)^FIELDS}')
    if config['model_variant'] not in VARIANTS or config['task'] not in ('USC12K','SOD','COD','JOINT'):
        raise ValueError('Unknown variant/task')
    for field in ('epochs','batch_size','image_size','expected_train_images','expected_steps_per_epoch'):
        if not isinstance(config[field],int) or config[field]<1: raise ValueError(f'Invalid {field}')
    if config['image_size']%32 or config['workers']<0: raise ValueError('Invalid image size/workers')
    expected='sod_anchored_pairs' if config['task']=='JOINT' else 'shuffle_drop_last'
    if config['sampling']!=expected or config['selection']!='minimum_training_objective':
        raise ValueError('Unsupported sampling/checkpoint selection')
    if config['task']=='JOINT' and config['batch_size']%2: raise ValueError('Joint batch must be even')
    return config

def make_dataset(config,args,training=True):
    task,size=config['task'],config['image_size']
    if task=='USC12K':
        if not args.usc_root: raise ValueError('--usc_root is required')
        return USC12KDataset(args.usc_root,'train',size,training=training)
    if task=='JOINT':
        if not args.sod_train or not args.cod_train: raise ValueError('Joint data require both training roots')
        return JointTaskDataset(args.sod_train,args.cod_train,size,training)
    root=args.sod_train if task=='SOD' else args.cod_train
    if not root: raise ValueError(f'Missing {task} training root')
    return BinaryTaskDataset(root,size,task,training)

def make_loader(config,dataset,generator,workers=None):
    count=config['workers'] if workers is None else workers
    common={'num_workers':count,'generator':generator,'worker_init_fn':seed_worker,'pin_memory':True}
    if config['task']=='JOINT':
        loader=DataLoader(dataset,batch_sampler=PairedBatchSampler(len(dataset.sod_samples),config['batch_size'],generator),**common)
    else:
        loader=DataLoader(dataset,batch_size=config['batch_size'],shuffle=True,drop_last=True,**common)
    if len(dataset)!=config['expected_train_images'] or len(loader)!=config['expected_steps_per_epoch']:
        raise ValueError(f'Data/budget mismatch: {len(dataset)} images, {len(loader)} steps')
    return loader

def data_identity(dataset,args,config):
    if config['task']=='USC12K':
        from .data_audit import audit_usc12k
        return {'kind':'USC12K','sample_count':len(dataset),**audit_usc12k(args.usc_root)}
    rows=[[a.name,a.stat().st_size,b.name,b.stat().st_size] for a,b in dataset.samples]
    return {'kind':config['task'],'sample_count':len(dataset),
            'relative_names_and_file_sizes_sha256':canonical_hash(rows),
            'fingerprint_scope':'Names and file sizes; not a content hash of every image'}
