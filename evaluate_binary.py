"""Evaluate both foreground outputs on named classic datasets at native GT size."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
from datasets import ClassicEvalDataset
from utils.checkpoint import load_model
from utils.foreground_metrics import ForegroundMetrics
from utils.runtime import atomic_save,resolve_device

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--datasets',required=True,type=Path,help='JSON object mapping dataset names to Imgs/GT roots')
    p.add_argument('--output',required=True,type=Path);p.add_argument('--device',default='auto')
    p.add_argument('--image_size',type=int,default=None)
    args=p.parse_args();device=resolve_device(args.device)
    if device.type=='cpu':torch.set_num_threads(2)
    model,state,identity=load_model(args.checkpoint,device)
    size=args.image_size or state['image_size'];del state
    roots=json.loads(args.datasets.read_text(encoding='utf-8-sig'))
    if not isinstance(roots,dict) or not roots:raise ValueError('Dataset mapping must be a nonempty JSON object')
    results={}
    with torch.inference_mode():
        for name,root in roots.items():
            dataset=ClassicEvalDataset(root,size)
            scores={h:{mode:ForegroundMetrics(mode=='png_minmax') for mode in ('raw','png_minmax')} for h in ('P0','Pu')}
            for sample in tqdm(dataset,desc=name,mininterval=10):
                output=model(sample['image'][None].to(device));target=sample['target'].numpy()
                # Resize logits, then sigmoid: do not resize probabilities.
                logits={'P0':output['base_final'],'Pu':output['union']}
                for head,logit in logits.items():
                    value=F.interpolate(logit,size=target.shape,mode='bilinear',align_corners=False).sigmoid()[0,0].cpu().numpy()
                    for acc in scores[head].values():acc.step(value,target,True)
            results[name]={'images':len(dataset),'heads':{h:{m:a.compute() for m,a in modes.items()} for h,modes in scores.items()}}
    atomic_save(args.output,{'identity':identity,'image_size':size,'results':results,
        'protocol':{'resize':'logit to native GT size, then sigmoid','primary_head':'Pu',
                    'raw':'float32 normalize=False','png_minmax':'floor to uint8, divide by255, per-image min-max; reported separately',
                    'head_selection':'Both heads on every dataset; no best-head selection'}})
    print(str(args.output.resolve()),flush=True)

if __name__=='__main__':main()
