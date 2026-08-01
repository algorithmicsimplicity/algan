import importlib
import os
import shutil
from unittest import TestCase

import cv2
import numpy as np
import torch
from parameterized import parameterized

from algan import *

current_file_path = os.path.abspath(__file__)
cd = os.path.dirname(current_file_path)
test_file_dir = os.path.join(cd, "test_files")
test_files = [[f] for f in os.listdir(test_file_dir) if f.endswith(".py")]

rendering_device = "cuda" if torch.cuda.is_available() else "cpu"


SETTINGS.style.fade_out_on_scene_end = True
SETTINGS.video.set(PREVIEW)
# The render device is chosen at import time from ALGAN_RENDER_DEVICE; there is
# no runtime setting for it (the old assignment here silently did nothing).
SETTINGS.paths.output_root = cd
# Keep the historical name for no-argument renders so the checked-in baselines
# in expected_outputs_* keep matching.
SETTINGS.paths.output_filename = "algan_render_output"
# Tests wipe the cache per-test (setUp) for hermetic renders. Point it at a
# test-local directory so the user's shared home cache is never deleted.
# ``taichi_cache_directory`` is deliberately NOT redirected: compiled kernels
# are content-independent and cost minutes to rebuild.
SETTINGS.paths.cache_directory = os.path.join(cd, "algan_cache")


class TestOverseer(TestCase):
    def setUp(self):
        if os.path.exists(SETTINGS.paths.cache_directory):
            shutil.rmtree(SETTINGS.paths.cache_directory)

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
                fps = y.get(cv2.CAP_PROP_FPS) or 30
                overall_max_diff = 0
                diff_frames = []
                frame_count_mismatch = None
                while True:
                    ret1, yh_ = yh.read()
                    ret2, y_ = y.read()
                    if not (ret1 and ret2):
                        if ret1 != ret2:
                            frame_count_mismatch = (ret1, ret2)
                        break
                    diff = np.abs(yh_.astype(np.int16) - y_.astype(np.int16)).astype(
                        np.uint8
                    )
                    diff_frames.append(diff)
                    overall_max_diff = max(overall_max_diff, int(diff.max()))
                yh.release()
                y.release()

                if overall_max_diff > 2:
                    error_dir = os.path.join(cd, "output_errors", module_name)
                    os.makedirs(error_dir, exist_ok=True)
                    error_path = os.path.join(error_dir, f)
                    h, w = diff_frames[0].shape[:2]
                    writer = cv2.VideoWriter(
                        error_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (w, h),
                    )
                    for diff in diff_frames:
                        writer.write(diff)
                    writer.release()

                if frame_count_mismatch is not None:
                    assert frame_count_mismatch[0] == frame_count_mismatch[1], (
                        f"{module_name} output does not have the expected number of frames."
                    )
                assert overall_max_diff <= 2, (
                    f"{module_name} output does not match expectation. Max pixel difference: {overall_max_diff}"
                )
                # with open(os.path.join(test_output_dir, f), 'r') as yh, open(os.path.getsize(), 'r') as y:
                # self.assertEqual(os.path.getsize(os.path.join(test_output_dir, f)), os.path.getsize(os.path.join(expected_output_dir, f)))
