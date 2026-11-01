"""Split/data integrity checks shared by preflight and reproducibility audits."""
from pathlib import Path
from dataset_usc12k import USC12KDataset
from .runtime import sha256

SCENES=('Scene-A','Scene-B','Scene-C','Scene-D')

def split_fingerprints(root):
    lists=Path(root)/'ImageSets/Segmentation'
    return {name:sha256(lists/f'{name}.txt') for name in ('train','val',*SCENES) if (lists/f'{name}.txt').is_file()}

def audit_usc12k(root,standard=True):
    ids={name:USC12KDataset(root,name,352,False).ids for name in ('train','val',*SCENES)}
    counts={name:len(items) for name,items in ids.items()}
    if standard and counts!={'train':8400,'val':3600,**{name:900 for name in SCENES}}:
        raise ValueError(f'Unexpected USC12K counts: {counts}')
    if set(ids['train'])&set(ids['val']):raise ValueError('Train and validation overlap')
    combined=[sid for name in SCENES for sid in ids[name]]
    if len(combined)!=len(set(combined)) or set(combined)!=set(ids['val']):
        raise ValueError('Scene lists must partition validation exactly')
    return {'counts':counts,'split_sha256':split_fingerprints(root),
            'all_paths_exist':True,'train_val_disjoint':True,'scene_partition_valid':True}
