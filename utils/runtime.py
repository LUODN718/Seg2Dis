"""Portable runtime, atomic output and content fingerprints."""
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]

def sha256(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): result.update(block)
    return result.hexdigest()

def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def source_manifest():
    paths=list(ROOT.glob('*.py'))
    for name in ('models','utils','datasets','scripts'):
        paths.extend((ROOT/name).rglob('*.py'))
    return {p.relative_to(ROOT).as_posix():sha256(p) for p in sorted(paths)}

def atomic_save(path, value, tensor=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    handle,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    os.close(handle)
    try:
        if tensor: torch.save(value,name)
        else: Path(name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
        os.replace(name,path)
    finally:
        if Path(name).exists(): Path(name).unlink()

def resolve_device(name):
    if name=='auto': name='cuda:0' if torch.cuda.is_available() else 'cpu'
    device=torch.device(name)
    if device.type=='cuda':
        if not torch.cuda.is_available(): raise RuntimeError('CUDA was requested but is unavailable')
        torch.cuda.set_device(device)
    return device

def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False

def seed_worker(worker_id):
    seed=torch.initial_seed()%2**32
    np.random.seed(seed);random.seed(seed)

def runtime_versions():
    import platform
    from importlib.metadata import version
    return {'python':platform.python_version(),
            **{name:version(name) for name in ('torch','torchvision','timm','numpy','Pillow','pysodmetrics','scipy')},
            'torch_cuda':torch.version.cuda}
