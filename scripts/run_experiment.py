"""Train one fixed current experiment, then verify all requested evaluations."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.experiment import load_config,make_dataset,data_identity
from utils.runtime import ROOT,atomic_save,sha256,source_manifest,canonical_hash,runtime_versions

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def call(root,name,arguments):
    atomic_save(root/'status.json',{'stage':name,'command':arguments})
    with (root/f'{name}.stdout.log').open('a',encoding='utf-8') as out, (root/f'{name}.stderr.log').open('a',encoding='utf-8') as err:
        result=subprocess.run([sys.executable,'-u',*map(str,arguments)],cwd=ROOT,stdout=out,stderr=err)
    if result.returncode:raise RuntimeError(f'{name} exited with code {result.returncode}; inspect its logs')

def run(args):
    config=load_config(args.config);root=args.output
    root.mkdir(parents=True,exist_ok=True)
    source=canonical_hash(source_manifest())
    dataset=make_dataset(config,args,training=False)
    expected={'config':config,'data_identity':data_identity(dataset,args,config),
              'runtime_versions':runtime_versions(),
              'source_fingerprint':source,'backbone_sha256':sha256(args.backbone_checkpoint),
              'classic_datasets':read(args.classic_datasets) if args.classic_datasets else None}
    del dataset
    if config['task']!='USC12K' and not args.classic_datasets:
        raise ValueError('Task experiments require an explicit classic dataset mapping')
    manifest=root/'run_manifest.json'
    if manifest.exists() and read(manifest)!=expected:raise ValueError('Existing experiment manifest differs')
    atomic_save(manifest,expected)
    if (root/'complete.json').exists():
        done=read(root/'complete.json')
        if done.get('source_fingerprint')!=source or done.get('checkpoint_sha256')!=sha256(root/'training/best.pth'):
            raise ValueError('Completed experiment identity differs')
        for name in ('metrics.json','summary.json','training/complete.json'):
            if not (root/name).is_file():raise ValueError('Completed experiment is missing '+name)
        metrics=read(root/'metrics.json')
        if metrics.get('images')!=3600 or metrics['identity']['checkpoint_sha256']!=done['checkpoint_sha256']:
            raise ValueError('Completed experiment has incomplete/mismatched USC evaluation')
        if config['model_variant']!='flr' and not (root/'saved_metrics.json').is_file():
            raise ValueError('Completed experiment lacks saved-prediction verification')
        if args.classic_datasets and not (root/'classic_metrics.json').is_file():
            raise ValueError('Completed experiment lacks classic evaluation')
        print('Already complete:',root,flush=True);return
    training=root/'training'
    command=[ROOT/'train.py','--config',args.config,'--backbone_checkpoint',args.backbone_checkpoint,
             '--output',training,'--device',args.device]
    for key in ('usc_root','sod_train','cod_train'):
        if getattr(args,key):command+=['--'+key,getattr(args,key)]
    if args.resume:command+=['--resume',training/'latest.pth']
    elif (training/'latest.pth').exists():raise ValueError('Existing training requires explicit --resume')
    if not (training/'complete.json').exists():call(root,'training',command)
    done=read(training/'complete.json')
    if not done.get('complete') or done['epochs']!=config['epochs']:raise ValueError('Training budget incomplete')
    provenance=read(training/'training_config.json')
    for key,value in (('training_config',config),('data_identity',expected['data_identity']),
                      ('source_fingerprint',source),('backbone_sha256',expected['backbone_sha256']),
                      ('runtime_versions',expected['runtime_versions'])):
        if provenance.get(key)!=value:raise ValueError('Training provenance differs: '+key)
    checkpoint=training/'best.pth'
    digest=sha256(checkpoint)
    if digest!=done['best_checkpoint_sha256']:raise ValueError('Best checkpoint hash differs')
    call(root,'usc_evaluation',[ROOT/'evaluate.py','--checkpoint',checkpoint,'--usc_root',args.usc_root,
         '--split','val','--output',root/'metrics.json','--predictions',root/'predictions',
         '--probabilities',root/'foreground_probabilities','--device',args.device])
    metrics=read(root/'metrics.json')
    if metrics['images']!=3600 or set(metrics['scenes'])!={'Scene-A','Scene-B','Scene-C','Scene-D'}:
        raise ValueError('USC12K evaluation is incomplete')
    if metrics['identity']['run_kind']!='training' or metrics['identity']['checkpoint_sha256']!=digest:
        raise ValueError('Evaluation identity is not the trained checkpoint')
    if len(list((root/'foreground_probabilities').rglob('*.npz')))!=3600:
        raise ValueError('Expected one raw foreground probability file per validation image')
    if config['task']=='USC12K' and metrics['dataset_split_sha256']!=expected['data_identity']['split_sha256']:
        raise ValueError('Evaluation split lists differ from the training data audit')
    if config['model_variant']!='flr':
        call(root,'saved_prediction_verification',[ROOT/'evaluate_predictions.py','--usc_root',args.usc_root,
             '--pred_root',root/'predictions','--format','color','--split','val','--output',root/'saved_metrics.json'])
        saved=read(root/'saved_metrics.json')
        if saved['overall']!=metrics['overall'] or saved['scenes']!=metrics['scenes']:
            raise ValueError('Direct and saved-prediction metrics differ')
    else:
        fg=metrics['foreground']
        for scene,value in fg['scenes'].items():
            if value['P0']!=value['Pu']:raise ValueError('Independent FLR has unequal P0/Pu metrics: '+scene)
    if args.classic_datasets:
        call(root,'classic_evaluation',[ROOT/'evaluate_binary.py','--checkpoint',checkpoint,
             '--datasets',args.classic_datasets,'--output',root/'classic_metrics.json','--device',args.device])
        classic=read(root/'classic_metrics.json')
        if set(classic['results'])!=set(expected['classic_datasets']) or classic['identity']['checkpoint_sha256']!=digest:
            raise ValueError('Classic evaluation is incomplete or uses another checkpoint')
    atomic_save(root/'summary.json',{'complete':True,'identity':metrics['identity'],
                                   'overall':metrics['overall'],'scenes':metrics['scenes'],
                                   'foreground':metrics['foreground']})
    atomic_save(root/'complete.json',{'complete':True,'architecture':'segdis_rgb_v2',
        'experiment_id':config['id'],'model_variant':config['model_variant'],'source_fingerprint':source,
        'checkpoint_sha256':digest,'training_config':config,
        'usc_images':3600,'classic_datasets':list(expected['classic_datasets'] or {})})
    atomic_save(root/'status.json',{'stage':'complete','experiment_id':config['id'],'checkpoint_sha256':digest})

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--config',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--backbone_checkpoint',required=True,type=Path);p.add_argument('--usc_root',required=True,type=Path)
    p.add_argument('--sod_train',type=Path);p.add_argument('--cod_train',type=Path)
    p.add_argument('--classic_datasets',type=Path);p.add_argument('--device',default='cuda:0')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    try:run(args)
    except Exception as error:
        atomic_save(args.output/'status.json',{'stage':'failed','error':repr(error)})
        raise

if __name__=='__main__':main()
