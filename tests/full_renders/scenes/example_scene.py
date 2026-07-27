from algan import *

mobs = Group([Square(), Circle(),
              Sphere(), Cylinder()]).arrange_in_grid().spawn()#.fit_to_screen_rectangle().spawn()

with Sync(run_time=3):
    mobs.rotate(180, UP)#.scale(0.75)
    for mob in mobs:
        mob.rotate(180, RIGHT)#.move((ORIGIN-mob.location) * 0.2)
