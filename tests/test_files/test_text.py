from algan import *


def text():
    x = Text("Hello", font_size=100).spawn()
    #p = x.character_mobs[0].location
    c1 = x.character_mobs[0]
    with Seq():
        c1.move(UP*0.1)
        c1.move(RIGHT* 0.1)
    return

def tex():
    x = Tex("\\int_{x=0}^1 World", font_size=100).spawn()
    # p = x.character_mobs[0].location
    c1 = x.character_mobs[0]
    with Seq():
        c1.move(UP * 0.1)
        c1.move(RIGHT * 0.1)
    return


render_all_funcs(__name__)
