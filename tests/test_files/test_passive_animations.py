from algan import *

def d():
    square = Square().spawn()
    # Set square rotating at a rate of 180 degrees per second, indefinitely.
    # The apply_passive_animation function returns the id the animation it creates,
    # we will need to hang onto it if we want to stop the animation later.
    passive_id_1 = square.add_updater(lambda self, t: self.rotate(t*180, OUT))

    square2 = Square(color=BLUE).move(RIGHT*1.5).spawn()
    # Make square2 track square's right direction.
    # Note that even though we don't use the t parameter here,
    # we still must declare it in the function signature.
    passive_id_2 = square2.add_updater(lambda self, t: self.move_to(square.location +
                                                                                square.get_right_direction()*1.5))

    # Now we can continue animating as usual, these passive animations will continue.
    square.wait(2)
    square.color = GREEN
    square.wait(2)

    # And we can stop the animations when we want.
    square2.remove_updater(passive_id_2)
    square.wait(2)

render_all_funcs(__name__, render_settings=PREVIEW, start_index=0, max_rendered=1)
