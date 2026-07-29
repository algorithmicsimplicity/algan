# Test Suite

Algan's tests are split by purpose:

- `unit_tests/` contains focused behavioral, geometry, timeline, memory, material,
  importer, and renderer assertions.
- `full_renders/scenes/` contains broad import-only Algan scenes.
- `full_renders/test_full_renders.py` renders every scene at `PREVIEW` quality
  and compares every decoded frame with a device-specific baseline.

Run the unit suite:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit_tests -q
```

Run the full-render suite:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\full_renders -q
```

After intentionally changing rendering, review the generated videos and
checkpoint PNGs, then update baselines:

```powershell
$env:ALGAN_UPDATE_FULL_RENDER_BASELINES = "1"
.\.venv\Scripts\python.exe -m pytest tests\full_renders -q
Remove-Item Env:\ALGAN_UPDATE_FULL_RENDER_BASELINES
```

## Full-Render Coverage

| Scene | Coverage |
| --- | --- |
| `timeline_and_text.py` | `Text`, `Tex`, `NumericDisplay`, 2D shapes, `Group`, `Seq`, `Sync`, `Lag`, `Off`, lifecycle, `become`, hierarchy transforms, updaters, indications |
| `materials_and_lighting.py` | all nine mesh material families, animated material parameters, ambient/directional lights, shadows, camera orbit |
| `geometry_and_camera.py` | core surfaces, closed cylinders/cones, torus, cube, polyhedra, `Arrow3D`, parent/child transforms, camera motion |
| `media_and_shaders.py` | `ImageMob`, texture interpolation, composed fragment shaders, GLB import, PBR/normal-map model rendering |

The media scene uses `full_renders/assets/textured_icosphere.glb`, a compact
UV-mapped GLB with embedded albedo, metallic/roughness, and normal textures.
Keeping this fixture intentionally small lets the render remain inside the
configured arena without weakening importer or material coverage.

## Audit Decisions

- The useful pytest modules formerly at `tests/` were moved to `unit_tests/`.
- `test_fragment_shaders.py` and `test_surface_autotune.py` were rescued from
  the old excluded `test_files/` directory.
- The removed time-compression test targeted a module that no longer exists.
- Raster-vs-ray, renderer-enable, and standalone demo scripts targeted deleted
  APIs. Their current guarantees are covered by the PN, BVH, material, shader,
  geometry, and full-render tests.
- Focused visual scripts under `test_files/` were consolidated into the four
  broad scenes above. The structure test enforces the import-only scene
  contract and audited feature matrix.
