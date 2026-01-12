"""CharacterData module - MakeHuman data and utilities for VNCCS Character Studio."""

from .mh_parser import TargetParser, HumanSolver
from .obj_loader import Mesh, load_obj
from .mesh_processing import subdivide_catmull_clark_approx
from .mh_skeleton import Skeleton, Bone, VertexBoneWeights

__all__ = [
    'TargetParser',
    'HumanSolver', 
    'Mesh',
    'load_obj',
    'subdivide_catmull_clark_approx',
    'Skeleton',
    'Bone',
    'VertexBoneWeights',
]
