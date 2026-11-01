"""Export real P0/Pu and type maps from a strictly identified current checkpoint."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from datasets.binary import image_tensor
from infer import save_prediction
from utils.checkpoint import load_model
from utils.runtime import atomic_save,resolve_device

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint',required=True,type=Path);p.add_argument('--image',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--image_size',type=int,default=352)
    p.add_argument('--device',default='auto')
    args=p.parse_args();device=resolve_device(args.device)
    if device.type=='cpu':torch.set_num_threads(2)
    model,state,identity=load_model(args.checkpoint,device);del state
    image=Image.open(args.image).convert('RGB').resize((args.image_size,args.image_size),Image.Resampling.BILINEAR)
    args.output.mkdir(parents=True,exist_ok=True);image.save(args.output/'input.png')
    with torch.inference_mode():out=model(image_tensor(image)[None].to(device))
    maps={head:out[head][0,0].cpu().numpy() for head in ('P0','Pu')}
    if 'type_probability' in out:
        maps.update(type_SOD=out['type_probability'][0,0].cpu().numpy(),type_COD=out['type_probability'][0,1].cpu().numpy())
    np.savez_compressed(args.output/'probabilities.npz',**maps)
    for name,value in maps.items():
        Image.fromarray(np.round(np.clip(value,0,1)*255).astype(np.uint8)).save(args.output/f'{name}.png')
    if 'three_class_probability' in out:save_prediction(out['three_class_probability'][0],'prediction',args.output,'color')
    atomic_save(args.output/'identity.json',{'identity':identity,'image':args.image.name,
        'probabilities':'NPZ contains raw floats; PNGs are visualization only','image_size':args.image_size})

if __name__=='__main__':main()
