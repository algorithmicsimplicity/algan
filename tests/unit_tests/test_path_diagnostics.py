from pathlib import Path

import pytest

from algan.errors import AlganConfigurationError
from algan.utils import path_utils


def test_write_probe_creates_directory_and_leaves_no_files(tmp_path):
    target = tmp_path / "cache"
    assert (
        path_utils._ensure_writable_directory(
            target, purpose="cache", remedy="Choose a directory."
        )
        == target
    )
    assert list(target.iterdir()) == []


def test_native_writer_gets_actionable_cache_error(monkeypatch, tmp_path):
    import algan.mobs.text as text
    from algan import SETTINGS

    attempted = []
    original_open = Path.open

    def denied(path, *args, **kwargs):
        if path.name.startswith(".algan-write-check-"):
            attempted.append(path)
            raise PermissionError("sandbox denied access")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", denied)
    with (
        SETTINGS.paths.override(cache_directory=str(tmp_path)),
        pytest.raises(AlganConfigurationError) as error,
    ):
        text.make_manim_dir()
    message = str(error.value)
    assert str(tmp_path / "manim" / "Tex") in message
    assert "SETTINGS.paths.cache_directory" in message
    assert "sandbox denied access" in message
    assert len(attempted) == 1


def test_existing_file_instead_of_directory_is_actionable(tmp_path):
    target = tmp_path / "file"
    target.write_text("keep")
    with pytest.raises(AlganConfigurationError, match="Cannot write speech"):
        path_utils._ensure_writable_directory(
            target, purpose="speech", remedy="Set ALGAN_CACHE_DIR."
        )
    assert target.read_text() == "keep"
