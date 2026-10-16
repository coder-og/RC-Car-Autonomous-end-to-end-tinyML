import tempfile
from pathlib import Path
import unittest
import numpy as np
import tensorflow as tf
import params
from train import iter_run_dirs, weighted_huber

class TrainingTests(unittest.TestCase):
    def test_scalar_loss_weights_each_sample(self):
        y = tf.constant([0.0, 1.0])
        pred = tf.constant([[0.0], [0.0]])
        self.assertAlmostEqual(float(weighted_huber(y, pred)), 0.285, places=6)
        self.assertAlmostEqual(float(weighted_huber(tf.reshape(y, (2,1)), pred)), 0.285, places=6)
    def test_held_out_run_is_excluded(self):
        previous = params.validation_run_dir
        try:
            with tempfile.TemporaryDirectory() as directory:
                train = Path(directory) / 'run_001'; train.mkdir()
                val = Path(directory) / 'run_002'; val.mkdir()
                params.validation_run_dir = str(val)
                self.assertEqual(iter_run_dirs(directory), [train])
        finally:
            params.validation_run_dir = previous
if __name__ == '__main__':
    unittest.main()
