import os, sys, runpy
os.environ["ALGAN_WIDE_ATTR_RENDER_DEVICE"] = "0"
sys.argv = ["determinism.py", "UHD", "2", "cputex"]
import algan
from algan import SETTINGS
SETTINGS.computing.set(max_cpu_memory_used=20_000_000_000)
runpy.run_path("scratch_perf/determinism.py", run_name="__main__")
