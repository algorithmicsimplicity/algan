from algan.animation.animation_contexts import Sync, Off, Seq
from algan.constants.spatial import *#RIGHT, LEFT, IN, OUT, ORIGIN, UP
from algan.mobs.bezier_circuit import BezierCircuitCubic
from algan.utils.algan_utils import render_all_funcs

p = torch.stack((
    torch.stack((UP, UP+RIGHT*0.1, UP+RIGHT*0.2, UP+RIGHT*-0.2+DOWN*0.5)),
    torch.stack((UP+RIGHT*-0.2+DOWN*0.5, UP+RIGHT*-0.3+DOWN*0.4, UP+RIGHT*-0.2+DOWN*0.1, UP)),
))

p = p - p.mean((0,1))
p = p * 4

get_mob = lambda q=p: BezierCircuitCubic(q, color=YELLOW, add_texture_grid=False, render_with_distance_to_curve=False).spawn()

#TODO at end of rotation the curve flips inside ot
#could this be because we just default normals to camera direction?
#and after the turn there's a batch break, the previous patch had normals that started facing camera then rotated,
#but the second batch now has normals facing camera (even though the basis has ben flipeed).
#Figure out how to fix the sliver (maybe render in local space?)

def test_bezier_circuit_basic():
    with Off():
        x1 = get_mob().move(IN * 0.1)  # .move(UP*0.5)
        x2 = get_mob()#.move(UP*0.5)
        x2.color = PURE_RED
        x2.move(UP*0.2)
        #d = Group([Sphere(radius=0.05, location=l, constants=GREEN*0.8) for l in x2.children[0].location[0]]).spawn()
        #x2.rotate(-125, UP)
        #x2.rotate(-90, RIGHT)

    #x2.wait()
    #return
    #with Sequenced(run_time=10):
    #    x2.rotate(-180, UP)
    #return
    #x2.wait()
    """
    with Synchronized(same_run_time=True): #TODO with run_time=8 is crash, with run_time=3 is visual glitch with bezier curve leaking outside it's boundary
        #x2.wave_color(GREEN, direction=RIGHT)
        #TODO there is a bug where when the curve is directly inline with camera (width 1), it's height increases, the height should stay the same.
        x1.rotate(30, UP)
        x2.rotate(-180, UP)
    """
    r = 10
    with Sync(run_time=r): #TODO with run_time=8 is crash, with run_time=3 is visual glitch with bezier curve leaking outside it's boundary
        with Seq(run_time=r):
            x2.wave_color(PURE_BLUE, direction=RIGHT + UP)
            #d.wave_color(BLUE, direction=RIGHT + UP)
        #TODO there is a bug where when the curve is directly inline with camera (width 1), it's height increases, the height should stay the same.
        #x1.rotate(30, UP)
        with Seq(run_time=1):
            x1.move(RIGHT*0.3)
        with Seq(run_time=r):
            x2.rotate(-180, UP)
        with Seq(run_time=r):
            x2.move(RIGHT * 2)
    with Seq(run_time=r):
        x2.rotate(-180, RIGHT)
    with Seq(run_time=r):
        x2.rotate(-180, RIGHT+UP)



def test_bezier_circuit_complex():
    with Off():
        c = torch.stack((LEFT+DOWN*0.5, RIGHT*0.5+UP*0.3, LEFT*0.4+UP*0.33, LEFT*0.7+DOWN*0.7))
        #c = torch.stack((c, c.flip(0)+DOWN*0.2))
        c = c.unsqueeze(0)
        x = get_mob(c)
        #x = get_mob(p)
    #with Sequenced():
    #    x.location = x.location + torch.randn_like(x.location * 0.05)
    #    x.location = x.location + torch.randn_like(x.location * 0.05)
    x.wait(1)
    #x2.rotate(-45, UP)
    #x.move(RIGHT)
    #x.location = x.location + RIGHT
    return


render_all_funcs(__name__, start_index=0, max_rendered=1)
