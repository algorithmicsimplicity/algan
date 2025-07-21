# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Algan is a 3D animation engine for explanatory math videos, designed as a successor to Manim. It uses PyTorch as its backend for GPU-accelerated rendering and animation. The project creates precise programmatic animations for mathematical content, similar to those in AlgorithmicSimplicity videos.

## Common Development Commands

### Building and Publishing
- **Build package**: `uv build` (run from project root)
- **Publish to PyPI**: `uv publish` (after updating version in pyproject.toml)
- **Clear cache**: Run `clear_cache()` in Python or delete `algan_cache/` directory

### Testing
- **Run all tests**: `python tests/run_tests.py`
- **Run specific test**: `python -m pytest tests/test_files/test_basic.py`
- Test files are located in `tests/test_files/` and use parameterized testing
- Tests compare rendered video output frame-by-frame with expected outputs in `tests/expected_outputs/`

### Documentation
- **Build docs**: `python docs/make_and_open_docs.py`
- **Documentation source**: `docs/source/`
- Uses Sphinx with custom templates and configuration

### Code Quality
- **Linting**: Uses Ruff (configured in pyproject.toml)
- **Type checking**: Standard Python typing with torch tensors
- **Code formatting**: Black-compatible settings via Ruff

## Architecture Overview

### Core Components

**Scene Management**
- `Scene` class (`algan/scene.py`): Central rendering and animation coordinator
- `SceneManager`: Singleton pattern managing scene instances and memory
- Scenes contain actors (renderable objects), effects, and timeline management

**Animation System**
- `AnimationManager`: Global animation state and execution coordination
- `AnimationContext` classes (`algan/animation/animation_contexts.py`): Control animation timing, synchronization, and rate functions
- Key contexts: `Sync()`, `Seq()`, `Off()`, `AnimationContext(prevent_spawn=True)`

**Renderable Objects (Mobs)**
- `Mob` base class (`algan/mobs/mob.py`): Foundation for all animated objects
- Shape categories:
  - 2D shapes: `Circle`, `Square`, `Triangle`, `Polygon` etc. (`algan/mobs/shapes_2d.py`)
  - 3D shapes: `Cylinder`, `Sphere` (`algan/mobs/shapes_3d.py`)
  - Bezier circuits: `BezierCircuitCubic`, `BezierCurveCubic` (`algan/mobs/bezier_circuit.py`)
  - Text: `Text`, `Tex`, `OldTex` (`algan/mobs/text.py`)
  - Groups: `Group` for combining multiple objects

**Rendering Pipeline**
- `Camera` (`algan/rendering/camera.py`): 3D view transformation and projection
- Primitives (`algan/rendering/primitives/`): Low-level rendering objects (triangles, bezier circuits)
- Shaders (`algan/rendering/shaders/`): PBR (Physically Based Rendering) materials
- Post-processing: Bloom filters and other effects

**Memory Management**
- PyTorch-based with `torch.inference_mode()` for efficiency
- `ManualMemory` class for GPU memory management
- Automatic garbage collection and cache clearing

### Key Design Patterns

**Animation Execution**
```python
# Objects must be spawned before animation
obj = Circle().spawn()
obj.move(RIGHT)  # Animates movement

# Synchronization
with Sync():
    obj1.move(LEFT)
    obj2.move(UP)  # Both animate simultaneously
```

**Torch Integration**
- Heavy use of PyTorch tensors for all mathematical operations
- GPU acceleration through CUDA when available
- `torch.compile` wrapper for performance optimization (when supported)

**Manim Compatibility**
- Imports Manim for text rendering and some utilities
- `ManimMob` wrapper for using Manim objects within Algan

## File Structure Patterns

- `algan/`: Main package
- `algan/mobs/`: Renderable object classes
- `algan/animation/`: Animation timing and context management
- `algan/rendering/`: Camera, primitives, shaders, post-processing
- `algan/settings/`: Configuration defaults and render settings
- `algan/utils/`: Utilities for memory, tensors, animations, and Python helpers
- `tests/`: Comprehensive test suite with video comparison
- `docs/`: Sphinx documentation with examples

## Development Notes

**Dependencies**
- Core: PyTorch, torchvision, numpy, opencv-python
- Manim integration: Uses manim>=0.18.0 for text rendering
- Media: moviepy for video processing
- Additional: scipy, svgelements, specialized math libraries

**Testing Strategy**
- All test files in `tests/test_files/` follow pattern: import algan, define test functions, call `render_all_funcs(__name__)`
- Tests render videos and compare output frame-by-frame
- Use `PREVIEW` rendering mode for faster testing

**Memory Considerations**
- GPU memory management is critical for 3D rendering
- `OutOfRenderMemory` exceptions handled gracefully
- Scene reset and cache clearing between tests

**WSL Development**
- Special mount configuration for Windows Subsystem for Linux: `sudo mount -t drvfs D: /mnt/d -o metadata`