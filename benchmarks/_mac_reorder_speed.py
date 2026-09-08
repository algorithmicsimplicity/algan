"""Controlled CPU/MPS comparison on one VM with equal fixed arenas."""
import os
import subprocess
import sys
from pathlib import Path

if "--child" in sys.argv:
    import runpy
    device = sys.argv[sys.argv.index("--child") + 1]
    sys.argv = ["_mps_warm_regression.py", "2", "UHD", "900"]
    d = runpy.run_path("benchmarks/_mps_warm_regression.py", run_name="warm_probe")
    import algan.utils.memory_utils as mu
    real_init = mu.ManualMemory.__init__
    def fixed_arena(self, portion, *args, **kwargs):
        if portion > 0 and kwargs.get("managed", True):
            kwargs["num_bytes"] = 1720 * 2**20
        return real_init(self, portion, *args, **kwargs)
    mu.ManualMemory.__init__ = fixed_arena
    if device == "mps":
        from _mps_reorder_probe import reorder
        import algan.rendering.raytracing.tracer as tracer
        def repaired(self, active, perm):
            n = int(active.numel())
            reorder(active, perm, self.spare, n)
            self.current, self.spare = self.spare, self.current
            self.size = n
            return self.current[:n]
        tracer._ArenaRayCompactor.reorder = repaired
    d["main"]()
else:
    out = Path("algan_outputs/reorder_speed")
    out.mkdir(parents=True, exist_ok=True)
    for device in ("cpu", "mps"):
        env = dict(os.environ, ALGAN_RENDER_DEVICE=device,
                   ALGAN_ANIMATION_DEVICE="cpu", ALGAN_TORCH_COMPILE="0",
                   ALGAN_MPS_HOST_MEMORY_SHARE="0", ALGAN_USE_DAEMON="0")
        path = out / (device + ".txt")
        with path.open("w") as f:
            p = subprocess.Popen([sys.executable, __file__, "--child", device],
                                 env=env, stdout=f, stderr=subprocess.STDOUT)
            try:
                code = p.wait(timeout=800)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
                code = -9
        print("SPEED_ARM", device, code, flush=True)
        print(path.read_text(), flush=True)
        if code:
            raise SystemExit(code)
