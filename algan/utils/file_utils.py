from algan.settings._startup import _ANIMATION_DEVICE
from algan.constants.color import Color


def get_image(file_path):
    if isinstance(file_path, str):
        import torchvision  # deferred: ~0.2 s of import algan

        file_path = (
            torchvision.io.read_image(file_path)
            .to(_ANIMATION_DEVICE)
            .permute(1, 2, 0)
        )
        file_path = file_path.float() / 255
    return Color.add_defaults(file_path)
