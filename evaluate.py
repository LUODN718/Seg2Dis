"""Fixed USC12K three-class and raw P0/Pu binary evaluation."""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset_usc12k import USC12KDataset
from infer import save_prediction
from utils.checkpoint import load_model
from utils.evaluation import PROTOCOL
from utils.foreground_metrics import ForegroundMetrics,BINARY_PROTOCOL
from utils.metrics import SegmentationMetric
from utils.runtime import atomic_save,resolve_device
from utils.data_audit import split_fingerprints

SCENES=('Scene-A','Scene-B','Scene-C','Scene-D')

def evaluate_usc(model,root,split,device,batch_size=4,workers=0,predictions=None,probabilities=None):
    selected=SCENES if split=='val' else (split,)
    if split=='val':
        val=USC12KDataset(root,'val',352,False).ids
        scene_ids=[s for scene in selected for s in USC12KDataset(root,scene,352,False).ids]
        if len(scene_ids)!=len(set(scene_ids)) or set(scene_ids)!=set(val):
            raise ValueError('Scene lists must partition val exactly')
    has_types=model.variant!='flr'
    overall=SegmentationMetric();scenes={};foreground={}
    abc={head:ForegroundMetrics() for head in ('P0','Pu')}
    all_binary={head:ForegroundMetrics() for head in ('P0','Pu')}
    images=0
    model.eval()
    with torch.inference_mode():
        for scene in selected:
            dataset=USC12KDataset(root,scene,352,False)
            metric=SegmentationMetric();binary={head:ForegroundMetrics() for head in ('P0','Pu')}
            loader=DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=workers)
            for batch in tqdm(loader,desc=scene,mininterval=10):
                output=model(batch['image'].to(device))
                if has_types:
                    probability=output['three_class_probability']
                    if probability.shape[1:]!=(3,352,352):raise ValueError('Unexpected three-class shape')
                    if not torch.isfinite(probability).all():raise FloatingPointError('Nonfinite type probability')
                    metric.update(probability.argmax(1),batch['semantic'])
                for i,sid in enumerate(batch['sample_id']):
                    semantic=batch['semantic'][i].numpy();target=semantic>0
                    values={head:output[head][i,0].cpu().numpy() for head in ('P0','Pu')}
                    for head,value in values.items():
                        binary[head].step(value,target,scene!='Scene-D',semantic)
                        all_binary[head].step(value,target,False,semantic)
                        if scene!='Scene-D':abc[head].step(value,target,True,semantic)
                    if predictions is not None and has_types:
                        save_prediction(probability[i],sid,Path(predictions)/scene,'color')
                    if probabilities is not None:
                        folder=Path(probabilities)/scene;folder.mkdir(parents=True,exist_ok=True)
                        np.savez_compressed(folder/f'{sid}.npz',**values)
            if has_types:
                overall.confusion_matrix+=metric.confusion_matrix
                scenes[scene]={'images':len(dataset),**metric.compute()}
            else:scenes[scene]={'images':len(dataset),'three_class_metrics':None,'reason':'binary foreground model'}
            foreground[scene]={head:acc.compute() for head,acc in binary.items()}
            images+=len(dataset)
    return {'images':images,'split':split,'protocol':PROTOCOL,'dataset_split_sha256':split_fingerprints(root),
            'overall':overall.compute() if has_types else None,'scenes':scenes,
            'foreground':{'protocol':BINARY_PROTOCOL,'scenes':foreground,
                'all_selected_scenes':{h:a.compute() for h,a in all_binary.items()},
                'Scene-ABC':{h:a.compute() for h,a in abc.items() if a.count}}}

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint',required=True,type=Path);p.add_argument('--usc_root',required=True,type=Path)
    p.add_argument('--split',default='val',choices=('val',*SCENES))
    p.add_argument('--output',required=True,type=Path);p.add_argument('--device',default='auto')
    p.add_argument('--batch_size',type=int,default=4);p.add_argument('--workers',type=int,default=0)
    p.add_argument('--predictions',type=Path);p.add_argument('--probabilities',type=Path)
    args=p.parse_args();device=resolve_device(args.device)
    if device.type=='cpu':torch.set_num_threads(2)
    model,state,identity=load_model(args.checkpoint,device);del state
    result=evaluate_usc(model,args.usc_root,args.split,device,args.batch_size,args.workers,args.predictions,args.probabilities)
    result['identity']=identity;atomic_save(args.output,result)
    print(str(args.output.resolve()),flush=True)

if __name__=='__main__':main()
