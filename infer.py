"""Export USC12K predictions through the same strict model loader as evaluation."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset_usc12k import USC12KDataset
from utils.checkpoint import load_model
from utils.runtime import atomic_save,resolve_device

def save_prediction(probability,sample_id,output,mode):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    label=probability.argmax(0).detach().cpu().numpy().astype(np.uint8)
    if mode=='label': Image.fromarray(label).save(output/f'{sample_id}.png')
    elif mode=='color':
        palette=np.array([[0,0,0],[255,0,0],[0,255,0]],dtype=np.uint8)
        Image.fromarray(palette[label]).save(output/f'{sample_id}.png')
    else: raise ValueError(f'Unknown three-class export format: {mode}')

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint',required=True,type=Path);p.add_argument('--usc_root',required=True,type=Path)
    p.add_argument('--split',default='val',choices=('val','Scene-A','Scene-B','Scene-C','Scene-D'))
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--format',default='color',choices=('color','label','binary'))
    p.add_argument('--save_probabilities',action='store_true')
    p.add_argument('--device',default='auto')
    args=p.parse_args();device=resolve_device(args.device)
    if device.type=='cpu':torch.set_num_threads(2)
    model,state,identity=load_model(args.checkpoint,device);del state
    if model.variant=='flr' and args.format!='binary':
        raise ValueError('FLR has no type output; select --format binary')
    args.output.mkdir(parents=True,exist_ok=True)
    dataset=USC12KDataset(args.usc_root,args.split,352,training=False)
    with torch.inference_mode():
        for batch in tqdm(DataLoader(dataset,batch_size=1,shuffle=False,num_workers=0),desc=args.split):
            output=model(batch['image'].to(device));sid=batch['sample_id'][0]
            if not torch.isfinite(output['Pu']).all():raise FloatingPointError('Nonfinite foreground probability')
            if args.format=='binary':
                Image.fromarray((output['Pu'][0,0].cpu().numpy()>=0.5).astype(np.uint8)*255).save(args.output/f'{sid}.png')
            else: save_prediction(output['three_class_probability'][0],sid,args.output,args.format)
            if args.save_probabilities:
                folder=args.output/'probabilities';folder.mkdir(exist_ok=True)
                np.savez_compressed(folder/f'{sid}.npz',P0=output['P0'][0,0].cpu().numpy(),Pu=output['Pu'][0,0].cpu().numpy())
    atomic_save(args.output/'identity.json',{'identity':identity,'split':args.split,'format':args.format,
                'image_size':352,'images':len(dataset),'scene_remapping':False})

if __name__=='__main__':main()
