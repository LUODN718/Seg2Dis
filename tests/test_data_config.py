import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from PIL import Image
import torch
from datasets import BinaryTaskDataset,JointTaskDataset,PairedBatchSampler,pair_files
from utils.experiment import load_config
from utils.runtime import ROOT
from train import make_optimizer,compute_objective

class DataConfigTests(unittest.TestCase):
    def test_all_experiment_definitions(self):
        plan=json.loads((ROOT/'configs/experiments.json').read_text())
        self.assertEqual(sum(x['required'] for x in plan['experiments']),14)
        self.assertEqual(plan['priority_queue'],[
            'USC12K_SEGDIS','USC12K_PFG_HEAD','USC12K_SEMANTIC_PFG_HEAD',
            'USC12K_DECODER_HEAD','USC12K_SEMANTIC_HEAD',
            'USC12K_SEGDIS_NO_FBRC','USC12K_FLR'])
        for row in plan['experiments']:
            c=load_config(ROOT/row['config'])
            self.assertEqual(c['id'],row['id'])
            if c['task']=='JOINT':
                self.assertEqual((c['expected_train_images'],c['expected_steps_per_epoch']),(21106,2639))
            if c['task'] in ('SOD','COD','JOINT'):
                self.assertEqual((c['backbone_lr'],c['base_lr']),(1e-5,1e-4))

    def test_intact_pairs_and_partial_last_batch(self):
        batches=list(PairedBatchSampler(7,8,torch.Generator().manual_seed(42)))
        self.assertEqual([len(b) for b in batches],[8,6])
        self.assertEqual(sorted(i for b in batches for i in b if i%2==0),list(range(0,14,2)))
        for batch in batches:
            self.assertTrue(all(batch[i+1]==batch[i]+1 and batch[i]%2==0 for i in range(0,len(batch),2)))

    def test_binary_data_and_task_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            roots=[]
            for task,count in (('sod',3),('cod',2)):
                root=Path(folder)/task;roots.append(root)
                (root/'Imgs').mkdir(parents=True);(root/'GT').mkdir()
                for i in range(count):
                    Image.fromarray(np.zeros((32,32,3),np.uint8)).save(root/'Imgs'/f'{i}.jpg')
                    Image.fromarray(np.ones((32,32),np.uint8)).save(root/'GT'/f'{i}.png')
            joint=JointTaskDataset(*roots,64,False)
            self.assertEqual(len(joint),6)
            for i in range(6):
                sample=joint[i]
                self.assertEqual(sample['semantic'].unique().item(),i%2+1)
            (roots[0]/'GT'/'extra.png').write_bytes((roots[0]/'GT'/'0.png').read_bytes())
            with self.assertRaises(ValueError):pair_files(roots[0])

    def test_joint_objective_rejects_unpaired_batch(self):
        output={'union':torch.zeros(1,1,32,32,requires_grad=True)}
        batch={'semantic':torch.ones(1,32,32,dtype=torch.long),'task_id':torch.tensor([0])}
        with self.assertRaises(ValueError):compute_objective(output,batch,'flr','JOINT','cpu')

if __name__=='__main__':unittest.main()
