import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from utils.evaluation import read_prediction
from utils.metrics import SegmentationMetric


class EvaluationTests(unittest.TestCase):
    def test_scene_a_type_error_is_not_corrected(self):
        metric = SegmentationMetric()
        metric.update(torch.tensor([0, 2, 1]), torch.tensor([0, 1, 1]))
        result = metric.compute()
        self.assertEqual(result['confusion_matrix'], [[1,0,0],[0,1,1],[0,0,0]])
        self.assertAlmostEqual(result['cscs'], 0.5)
        self.assertAlmostEqual(result['iou'][1], 0.5)

    def test_empty_predictions_and_absent_classes(self):
        metric = SegmentationMetric()
        metric.update(torch.zeros(4), torch.zeros(4))
        result = metric.compute()
        self.assertIsNone(result['cscs'])
        self.assertEqual(result['iou'], [1.0, None, None])

    def test_verified_original_raw_matrix(self):
        metric = SegmentationMetric()
        metric.confusion_matrix = torch.tensor([[391066097,3044948,1984560],
            [3262107,28084205,1231025],[3341976,1347921,12691561]], dtype=torch.float64)
        result = metric.compute()
        self.assertAlmostEqual(result['fg_miou_percent'], 68.7913954714, places=8)
        self.assertAlmostEqual(result['cscs_percent'], 5.9445972403, places=8)

    def test_png_formats_and_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'sample.png'
            labels = np.zeros((352,352), np.uint8)
            labels[:,100:200] = 1; labels[:,200:] = 2
            Image.fromarray(labels).save(path)
            self.assertTrue(np.array_equal(read_prediction(path,'label'),labels))
            with self.assertRaises(ValueError): read_prediction(path,'color')
            palette = np.array([[0,0,0],[255,0,0],[0,255,0]], np.uint8)
            Image.fromarray(palette[labels]).save(path)
            self.assertTrue(np.array_equal(read_prediction(path,'color'),labels))
            with self.assertRaises(ValueError): read_prediction(path,'label')
            Image.fromarray(np.full((352,352),255,np.uint8)).save(path)
            with self.assertRaises(ValueError): read_prediction(path,'label')
            Image.fromarray(np.full((352,352,3),255,np.uint8)).save(path)
            with self.assertRaises(ValueError): read_prediction(path,'color')
            Image.fromarray(labels[:100]).save(path)
            with self.assertRaises(ValueError): read_prediction(path,'label')


if __name__ == '__main__': unittest.main()
