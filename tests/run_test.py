import torch
import cv2
import importlib
import numpy as np
import os
import shutil
from unittest import TestCase
from parameterized import parameterized

from algan import (
    PREVIEW,
    HD,
    RENDERING_DEFAULTS,
    STYLE_DEFAULTS,
    COMPUTING_DEFAULTS,
    DIRECTORY_DEFAULTS,
)

current_file_path = os.path.abspath(__file__)
cd = os.path.dirname(current_file_path)
test_file_dir = os.path.join(cd, "test_files")
test_files = [[f] for f in os.listdir(test_file_dir) if f.endswith(".py")]

rendering_device = "cuda" if torch.cuda.is_available() else "cpu"


STYLE_DEFAULTS.fade_out_on_scene_end = True
RENDERING_DEFAULTS.settings = PREVIEW
COMPUTING_DEFAULTS.render_device = torch.device(rendering_device)
DIRECTORY_DEFAULTS.base_directory = cd


class TestOverseer(TestCase):
    def setUp(self):
        if os.path.exists(DIRECTORY_DEFAULTS.cache_directory):
            shutil.rmtree(DIRECTORY_DEFAULTS.cache_directory)

    @parameterized.expand(test_files)
    def test_algan_file(self, test_file):
        module_name = os.path.splitext(test_file)[0]
        module_name = f"tests.test_files.{module_name}"

        importlib.import_module(module_name)

        test_output_dir = os.path.join(cd, "algan_outputs", module_name)
        expected_output_dir = os.path.join(
            cd, f"expected_outputs_{rendering_device}", module_name
        )
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
                        self.assertEqual(
                            ret1,
                            ret2,
                            f"{module_name} output does not have the expected number of frames.",
                        )
                        break
                    max_diff = np.abs(yh_ - y_).max()
                    self.assertLessEqual(
                        max_diff,
                        2,
                        f"{module_name} output does not match expectation. Max pixel difference: {max_diff}",
                    )
                yh.release()
                y.release()
                # with open(os.path.join(test_output_dir, f), 'r') as yh, open(os.path.getsize(), 'r') as y:
                # self.assertEqual(os.path.getsize(os.path.join(test_output_dir, f)), os.path.getsize(os.path.join(expected_output_dir, f)))
