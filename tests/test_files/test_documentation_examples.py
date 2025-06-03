from algan import *

parent_mob = Square(color=BLUE)
children_mobs = [Square(location=loc) for loc in [LEFT*1.5, UP*1.5, RIGHT*1.5, DOWN*1,5]]

parent_mob.add_children(children_mobs) # this is the crucial step

# Now, any change that we make to the parent mob will be propagated to the children mobs (including spawning).
parent_mob.spawn()

render_to_file()