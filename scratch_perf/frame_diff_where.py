import sys, cv2, numpy as np
a, b = sys.argv[1], sys.argv[2]
frames = [int(x) for x in sys.argv[3].split(",")]
ca, cb = cv2.VideoCapture(a), cv2.VideoCapture(b)
i = 0
while True:
    ra, fa = ca.read(); rb, fb = cb.read()
    if not ra or not rb: break
    if i in frames:
        d = np.abs(fa.astype(np.int16) - fb.astype(np.int16)).max(-1)
        ys, xs = np.nonzero(d > 2)
        h, w = d.shape
        print(f"frame {i}: {len(xs)} px differ; max {d.max()}; bbox x[{xs.min() if len(xs) else -1},{xs.max() if len(xs) else -1}] y[{ys.min() if len(ys) else -1},{ys.max() if len(ys) else -1}] of {w}x{h}")
        # coarse 8x8 grid histogram of differing pixels
        g = np.zeros((6, 8), int)
        for y, x in zip(ys[::max(1, len(ys)//20000)], xs[::max(1, len(xs)//20000)]):
            g[min(5, y*6//h), min(7, x*8//w)] += 1
        print(g)
        cv2.imwrite(f"scratch_perf/diff_frame{i}.png", np.clip(d * 3, 0, 255).astype(np.uint8))
    i += 1
