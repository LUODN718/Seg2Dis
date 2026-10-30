import tempfile
from pathlib import Path
import unittest
import numpy as np
from PIL import Image
import torch
from utils.metrics import SegmentationMetric,pixel_partition
from utils.evaluation import read_prediction
from utils.foreground_metrics import ForegroundMetrics
from infer import save_prediction

class EvaluationTests(unittest.TestCase):
    def test_scene_a_type_swap_is_preserved(self):
        metric=SegmentationMetric()
        metric.update(torch.tensor([0,2,1,0]),torch.tensor([0,1,1,1]))
        result=metric.compute();row=result['pixel_partition']['SOD']
        self.assertEqual(result['confusion_matrix'],[[1,0,0],[1,1,1],[0,0,0]])
        self.assertEqual((row['correct_pixels'],row['swapped_pixels'],row['missed_background_pixels']),(1,1,1))
        self.assertAlmostEqual(sum(row[k] for k in ('correct_percent','swapped_percent','missed_background_percent')),100)

    def test_overall_sums_counts_not_scene_scores(self):
        a,b=SegmentationMetric(),SegmentationMetric()
        a.update(torch.tensor([1]),torch.tensor([1]))
        b.update(torch.zeros(9),torch.ones(9))
        overall=SegmentationMetric();overall.confusion_matrix=a.confusion_matrix+b.confusion_matrix
        self.assertAlmostEqual(overall.compute()['iou'][1],0.1)

    def test_absent_classes_and_background_false_positives(self):
        metric=SegmentationMetric();metric.update(torch.tensor([0,1,2]),torch.zeros(3))
        result=metric.compute()
        self.assertEqual(result['cscs'],0)
        self.assertEqual(result['iou'][1:], [0,0])
        self.assertIsNone(result['pixel_partition']['SOD']['correct_percent'])
        metric=SegmentationMetric();metric.update(torch.zeros(4),torch.zeros(4))
        self.assertIsNone(metric.compute()['cscs'])

    def test_prediction_export_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            labels=torch.zeros(352,352,dtype=torch.long);labels[:,100:200]=1;labels[:,200:]=2
            probability=torch.nn.functional.one_hot(labels,3).permute(2,0,1).float()
            for mode in ('label','color'):
                save_prediction(probability,'sample',Path(folder),mode)
                self.assertTrue(np.array_equal(read_prediction(Path(folder)/'sample.png',mode),labels.numpy()))
            Image.fromarray(np.full((352,352),255,np.uint8)).save(Path(folder)/'sample.png')
            with self.assertRaises(ValueError):read_prediction(Path(folder)/'sample.png','label')
            Image.fromarray(np.zeros((64,64,3),np.uint8)).save(Path(folder)/'sample.png')
            with self.assertRaises(ValueError):read_prediction(Path(folder)/'sample.png','color')

    def test_binary_detection_is_not_type_correctness(self):
        metric=ForegroundMetrics()
        semantic=np.array([[1,2],[0,0]])
        metric.step(np.array([[0.8,0.7],[0.1,0.6]],np.float32),semantic>0,True,semantic)
        result=metric.compute()
        self.assertEqual(result['confusion_matrix'],[[1,1],[0,2]])
        self.assertEqual(result['region_recall']['COD']['recall'],1)
        self.assertNotIn('cscs',result)
        self.assertAlmostEqual(result['binary_foreground_IoU'],2/3)

if __name__=='__main__':unittest.main()
