"""Native binary data and intact SOD/COD pairs; no external repository imports."""
from pathlib import Path
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from config import CFG

EXTENSIONS={'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}

def pair_files(root):
    root=Path(root)
    def scan(folder):
        if not folder.is_dir(): raise FileNotFoundError(folder)
        result={}
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS: continue
            if path.stem in result: raise ValueError(f'Duplicate sample stem: {path}')
            result[path.stem]=path
        return result
    images,masks=scan(root/'Imgs'),scan(root/'GT')
    if not images or images.keys()!=masks.keys():
        raise ValueError(f'Empty or unpaired image/mask files in {root}')
    return [(images[k],masks[k]) for k in sorted(images)]

def read_mask(path):
    with Image.open(path) as f: mask=np.asarray(f.convert('L'),dtype=np.uint8).copy()
    if mask.max(initial=0)<=1: mask*=np.uint8(255)
    return mask

def image_tensor(image):
    x=torch.from_numpy(np.array(image,dtype=np.float32).transpose(2,0,1)/255)
    return (x-x.new_tensor(CFG.mean).view(3,1,1))/x.new_tensor(CFG.std).view(3,1,1)

def training_sample(pair,size,task_id,training):
    image_path,mask_path=pair
    with Image.open(image_path) as f: image=f.convert('RGB').resize((size,size),Image.Resampling.BILINEAR)
    mask=Image.fromarray(read_mask(mask_path)).resize((size,size),Image.Resampling.NEAREST)
    if training and random.random()<0.5:
        image=image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        mask=mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    target=torch.from_numpy((np.array(mask)>128).astype(np.int64))
    return {'image':image_tensor(image),'semantic':target*(task_id+1),
            'mask':target[None].float(),'task_id':torch.tensor(task_id),
            'sample_id':image_path.stem}

class BinaryTaskDataset(Dataset):
    def __init__(self,root,size,task,training=True):
        self.samples=pair_files(root)
        self.size,self.task_id,self.training=size,{'SOD':0,'COD':1}[task],training
    def __len__(self): return len(self.samples)
    def __getitem__(self,index):
        return training_sample(self.samples[index],self.size,self.task_id,self.training)

class JointTaskDataset(Dataset):
    def __init__(self,sod_root,cod_root,size,training=True):
        self.sod_samples,self.cod_samples=pair_files(sod_root),pair_files(cod_root)
        self.samples=self.sod_samples+self.cod_samples
        self.size,self.training=size,training
    def __len__(self): return 2*len(self.sod_samples)
    def __getitem__(self,index):
        task=index%2
        pool=self.cod_samples if task else self.sod_samples
        return training_sample(pool[(index//2)%len(pool)],self.size,task,self.training)

class PairedBatchSampler(Sampler):
    def __init__(self,pair_count,batch_size,generator):
        if pair_count<1 or batch_size<2 or batch_size%2:
            raise ValueError('Intact SOD/COD pairs require a positive pair count and even batch size')
        self.pair_count,self.pairs_per_batch,self.generator=pair_count,batch_size//2,generator
    def __len__(self): return (self.pair_count+self.pairs_per_batch-1)//self.pairs_per_batch
    def __iter__(self):
        order=torch.randperm(self.pair_count,generator=self.generator).tolist()
        for start in range(0,self.pair_count,self.pairs_per_batch):
            yield [i for p in order[start:start+self.pairs_per_batch] for i in (2*p,2*p+1)]

class ClassicEvalDataset(Dataset):
    def __init__(self,root,image_size=384):
        self.samples,self.image_size=pair_files(root),image_size
    def __len__(self): return len(self.samples)
    def __getitem__(self,index):
        image_path,mask_path=self.samples[index]
        with Image.open(image_path) as f:
            image=f.convert('RGB').resize((self.image_size,self.image_size),Image.Resampling.BILINEAR)
        target=read_mask(mask_path)>128
        return {'image':image_tensor(image),'target':torch.from_numpy(target.copy()),
                'sample_id':image_path.stem,'original_size':target.shape}
