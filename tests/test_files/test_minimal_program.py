from algan import *

group = Group([Square() for _ in range(3)]).arrange_in_line(RIGHT).spawn()
group.rotate(90, OUT)

Scene.save_video(reset=True)
