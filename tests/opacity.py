from algan import *
from algan.rendering.raytracing import enable_ray_tracing


def scene():
    spotlight = Circle(
        radius=10.0, color=GRAY, opacity=0.2, border_width=0
    )
    with Off():
        spotlight.spawn()
    spotlight.wait()

if __name__ == "__main__":
    enable_ray_tracing(1)
    render_all_funcs(__name__)