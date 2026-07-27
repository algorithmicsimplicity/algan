from algan import *

def get_mob():
    return Square()

xs = Group([get_mob() for _ in range(3)]).arrange_in_line().spawn()
with Sync(run_time=3):
    xs.rotate(90, OUT)
    [x.move(RIGHT * 2) for x in xs]

#Square().spawn().move_to(ORIGIN+LEFT*7)
#Circle().spawn().move_to(ORIGIN + LEFT * 3)