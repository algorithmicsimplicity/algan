import cv2
import importlib
import numpy as np
import os
from unittest import TestCase
from parameterized import parameterized

import algan.defaults.render_defaults
from algan import PREVIEW

test_file_dir = 'test_files'
test_files = [[f] for f in os.listdir(test_file_dir) if f.endswith('.py')]

algan.defaults.render_defaults.DEFAULT_RENDER_SETTINGS = PREVIEW

class TestOverseer(TestCase):
    def setUp(self):
        pass

    @parameterized.expand(test_files)
    def test_algan_file(self, test_file):
        module_name = os.path.splitext(test_file)[0]
        module_name = f'{test_file_dir}.{module_name}'

        importlib.import_module(module_name)
        test_output_dir = os.path.join('algan_outputs', module_name)
        expected_output_dir = os.path.join('expected_outputs', module_name)
        if os.path.exists(expected_output_dir):
            for f in os.listdir(expected_output_dir):
                if not os.path.exists(os.path.join(test_output_dir, f)):
                    continue
                yh = cv2.VideoCapture(os.path.join(test_output_dir, f))
                y = cv2.VideoCapture(os.path.join(expected_output_dir, f))
                while True:
                    ret1, yh_ = yh.read()
                    ret2, y_ = y.read()
                    if not (ret1 and ret2):
                        self.assertEqual(ret1, ret2, f'{module_name} output does not have the expected number of frames.')
                        break
                    self.assertEqual(np.abs(yh_ - y_).max(), 0, f'{module_name} output does not match expectation.')
                yh.release()
                y.release()
                #with open(os.path.join(test_output_dir, f), 'r') as yh, open(os.path.getsize(), 'r') as y:
                #self.assertEqual(os.path.getsize(os.path.join(test_output_dir, f)), os.path.getsize(os.path.join(expected_output_dir, f)))
