"""Importing 3-D model files.

:class:`~algan.mobs.three_d_models.model_mob.ThreeDModelMob` loads a ``.glb``,
``.gltf`` or ``.fbx`` file -- its meshes, its materials, and its rigid node
animation -- and presents it as an ordinary Algan Mob you can spawn, move and
light with everything else in the Scene.

:class:`~algan.mobs.three_d_models.mesh.TriangleMesh` is the underlying geometry
type for an explicit vertex/face mesh, useful on its own when you have the
triangles already. The ``scene_data`` types
(:class:`~algan.mobs.three_d_models.scene_data.SceneData` and friends) are the
format-neutral intermediate an importer produces, so support for a new file
format means writing one reader rather than touching the Mob layer.

See :doc:`/advanced_user_tutorials/three_d_models`.
"""

from __future__ import annotations

from algan.mobs.three_d_models.mesh import TriangleMesh, image_to_texture_map
from algan.mobs.three_d_models.model_mob import ThreeDModelMob
from algan.mobs.three_d_models.scene_data import (
    AnimationData,
    MaterialData,
    MeshData,
    NodeAnimation,
    NodeData,
    SceneData,
    SkinData,
)
