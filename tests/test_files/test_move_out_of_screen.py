from algan import *

def move_out_of_screen(mob_func):
    with Seq():
        for dir in [LEFT, RIGHT, UP, DOWN]:
            mob = mob_func().spawn()
            with Sync(run_time=3):
                mob.rotate(720, OUT)
                mob.move_out_of_screen(dir)

def test_move_out_of_screen_triangle():
    move_out_of_screen(Cylinder)

def test_move_out_of_screen_bezier():
    move_out_of_screen(Square)

render_all_funcs(__name__)