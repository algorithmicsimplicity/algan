import os, sys, runpy
os.environ["ALGAN_VIDEO_ENCODER"] = "software"
sys.argv = ["determinism.py", "UHD", "1", "batch3", "3"]
runpy.run_path("scratch_perf/determinism.py", run_name="__main__")
