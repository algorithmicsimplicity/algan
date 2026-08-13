Algan is an animation engine for explanatory math videos meant to supersede Manim. It's used to create
precise animations programmatically, as demonstrated in the videos of [AlgorithmicSimplicity](https://www.youtube.com/@algorithmicsimplicity).

## Table of Contents:

-  [Installation](#installation)
-  [Documentation](#documentation)
-  [License](#license)

## Installation


Algan requires a few dependencies that must be installed prior to using it. For detailed installation instructions,
please visit the [Documentation](https://algorithmicsimplicity.github.io/algan/installation/uv.html) and follow the appropriate instructions for your
operating system.

## Usage

Algan is an extremely versatile package. The following is an example `Scene` you can construct:

```python
from algan import *

circle = Circle(add_to_scene=False)
circle.set(border_color=PINK.set_opacity(0.5))
square = Square()
square.rotate(-135)  # Algan angles are in degrees, not radians

square.spawn()
square = square.become(circle)

Scene.save_video("example")
```

In order to view the output of this scene, save the code in a file called `example.py`. Then, run the following in a terminal window:

```sh
python example.py
```

You should see an algan_outputs directory in the same directory as your example.py file, containing `example.mp4`.

## Documentation

Documentation is in progress at [Documentation](https://algorithmicsimplicity.github.io/algan).

## License

The software is double-licensed under the MIT license, with copyright by Algorithmic Simplicity (see LICENSE).
