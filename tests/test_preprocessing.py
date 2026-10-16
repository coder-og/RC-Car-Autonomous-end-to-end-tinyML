import unittest
import numpy as np
import params
from preprocessing import preprocess_frame

class PreprocessingTests(unittest.TestCase):
    def test_crop_preserves_channels_and_size(self):
        image = np.zeros((240, 360, 3), dtype=np.uint8)
        image[params.crop_y1:params.crop_y2] = [10, 20, 30]
        result = preprocess_frame(image)
        self.assertEqual(result.shape, (params.inputres[1], params.inputres[0], 3))
        np.testing.assert_array_equal(result[0, 0], [10, 20, 30])
    def test_invalid_frame_rejected(self):
        for frame in (None, np.zeros((20, 20, 3)), np.zeros((240, 360))):
            with self.assertRaises(ValueError):
                preprocess_frame(frame)
if __name__ == '__main__':
    unittest.main()
