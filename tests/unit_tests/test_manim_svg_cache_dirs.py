import shutil
from pathlib import Path

from algan.settings import SETTINGS
from algan.utils.manim_svg_cache import _configure_manim_dirs


class _FakeManimConfig:
    def get_dir(self, name):
        return Path(getattr(self, name))


class _FakeTexTemplate:
    def get_texcode_for_expression(self, expression):
        return expression

    def get_texcode_for_expression_in_env(self, expression, environment):
        return expression


def test_configure_manim_dirs_creates_nested_cache_directories(tmp_path):
    cache_directory = tmp_path / "missing" / "nested" / "cache"
    config = _FakeManimConfig()

    with SETTINGS.paths.override(cache_directory=cache_directory):
        tex_dir, text_dir = _configure_manim_dirs(config)

    assert tex_dir == cache_directory / "manim" / "Tex"
    assert text_dir == cache_directory / "manim" / "texts"
    assert tex_dir.is_dir()
    assert text_dir.is_dir()


def test_vendored_generate_tex_file_recreates_directory_after_cache_wipe(tmp_path):
    from manim import config
    from algan.external_libraries.manim.utils import tex_file_writing

    old_tex_dir = config.tex_dir
    old_text_dir = config.text_dir
    cache_directory = tmp_path / "cache"
    try:
        with SETTINGS.paths.override(cache_directory=cache_directory):
            tex_dir, _ = _configure_manim_dirs(config)
            shutil.rmtree(cache_directory)

            tex_file = tex_file_writing.generate_tex_file(
                "x", tex_template=_FakeTexTemplate()
            )

        assert tex_file.parent == tex_dir
        assert tex_file.is_file()
    finally:
        config.tex_dir = old_tex_dir
        config.text_dir = old_text_dir
