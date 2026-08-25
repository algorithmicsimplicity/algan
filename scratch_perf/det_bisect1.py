import os, sys, runpy
os.environ["ALGAN_VIDEO_ENCODER"] = "software"
os.environ["ALGAN_WIDE_ATTR_RENDER_DEVICE"] = "0"
sys.argv = ["determinism.py", "UHD", "1", "bisect_widetexoff_b3", "3"]
runpy.run_path("scratch_perf/determinism.py", run_name="__main__")
