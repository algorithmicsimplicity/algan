from algan import Renderable


class PointLight(Renderable):
    def __init__(self, *args, **kwargs):
        kwargs['add_to_scene'] = False
        super().__init__(*args, **kwargs)