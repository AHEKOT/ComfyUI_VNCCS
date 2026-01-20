
import sys
import os
import json
import numpy as np

# Setup Path to include Project Root so we can import 'nodes' and 'CharacterData'
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

# MOCK ComfyUI API env (server, app, etc. if needed)
# The `__init__.py` imports `folder_paths` etc?
# Let's inspect `nodes/__init__.py` dependencies. 
# It imports `.character_creator` etc. which import `..utils`. 
# This implies we need to treat the root folder as a package `ComfyUI_VNCCS` or similar?
# OR we mock `..utils` if we import `nodes` top level.

# Strategy: Mock `nodes` package structure in sys.modules so relative imports work?
# No, simpler: Import `CharacterData` first so it's cached.
# Then try to import `nodes.ik_solver`.

import CharacterData.matrix # Ensure this works
import CharacterData.transformations # Ensure this works

# Creating a Mock Skeleton to pass to the handler
class MockBone:
    def __init__(self, name, head, tail, parent=None):
        self.name = name
        self.headPos = list(head)
        self.tailPos = list(tail)
        self.parent = parent
        self.matPose = np.asmatrix(np.identity(4))
        self.matPoseGlobal = np.asmatrix(np.identity(4))
        self.matPoseVerts = np.asmatrix(np.identity(4))
        self.matRestGlobal = np.asmatrix(np.identity(4))
        self.matRestGlobal[:3,3] = np.array(head).reshape(3,1) # Matrix assignment needs 2D or robust handling
        self.children = []
        if parent: parent.children.append(self)

    def update(self):
        if self.parent:
             self.matPoseGlobal = np.dot(self.parent.matPoseGlobal, self.matPose)
        else:
             self.matPoseGlobal = self.matPose
        # Simple skinning logic for test: M_global * inv(M_rest)
        self.matPoseVerts = np.dot(self.matPoseGlobal, np.linalg.inv(self.matRestGlobal))

class MockSkeleton:
    def __init__(self):
        self.bones = {}
    def getBone(self, name):
        return self.bones.get(name)
    def getBones(self):
        return list(self.bones.values())
    # No update() method, matching reality

# Define the Handler Logic (Inline or Imported?)
# The handler is `vnccs_character_studio_update_preview` in `nodes/__init__.py`.
# We want to import that function.
# `from nodes import vnccs_character_studio_update_preview` might fail due to other imports in `__init__`.
# `__init__` imports A LOT. `torch`, `PIL`, etc.
# If those are missing in this test env, it will fail.
# Assuming standard python env might not have `torch`.
# I'll TRY to import it. If it fails, I will copy-paste the logic of the handler for verification of the *Solver Call* specifically.
# But the user error was IN `ik_solver.py` via `solve_ik`.
# So testing `IKSolver.solve_ik` with the REAL module is the integrity check we need most.

def test_ik_integration():
    print("--- 1. Importing IKSolver directly ---")
    try:
        import importlib.util
        file_path = os.path.join(PROJECT_ROOT, 'nodes', 'ik_solver.py')
        spec = importlib.util.spec_from_file_location("ik_solver_module", file_path)
        ik_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ik_mod)
        IKSolver = ik_mod.IKSolver
        print("Success: IKSolver imported.")
    except Exception as e:
        print(f"FAILED to import IKSolver: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    print("--- 2. Setting up Skeleton ---")
    skel = MockSkeleton()
    b1 = MockBone("root", [0,0,0], [0,10,0])
    b2 = MockBone("mid", [0,10,0], [0,20,0], b1)
    b3 = MockBone("tip", [0,20,0], [0,21,0], b2)

    # initialize matPose with translations
    b1.matPose[:3,3] = np.array([0,0,0]).reshape(3,1)
    b2.matPose[:3,3] = np.array([0,10,0]).reshape(3,1)
    b3.matPose[:3,3] = np.array([0,10,0]).reshape(3,1)

    skel.bones = {"root": b1, "mid": b2, "tip": b3}
    
    # Run update once manually
    for b in skel.getBones(): b.update()
    
    target = [10, 10, 0] # 90 deg bend
    
    print("--- 3. Running solve_ik ---")
    try:
        # This will trigger 'skeleton.update()' (which we fixed to loop)
        # And 'import transformations' (which we fixed)
        rotations = IKSolver.solve_ik(skel, "tip", target, chain_length=2)
        print(f"Success: Solved. Rotations: {rotations}")
        
    except Exception as e:
        print(f"FAILED during solve_ik: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
        
    print("--- 4. Validating Result ---")
    # Apply rotations to check if it worked (Logic check)
    # The solver *applies* to bone.matPose internally too? 
    # Yes, `bone.matPose = ...` inside loop.
    # So `skel` state should be modified.
    
    for b in skel.getBones(): b.update()
    
    head_world_rest = np.array([0,20,0,1])
    # Tip Head is at 20 in rest.
    final_pos = np.asarray(np.dot(b3.matPoseVerts, head_world_rest)).ravel()[:3]
    print(f"Final Tip Pos: {final_pos}")
    
    dist = np.linalg.norm(final_pos - target)
    print(f"Distance: {dist}")
    if dist < 0.5:
        print("TEST PASSED")
    else:
        print("TEST FAILED: Convergence bad")
        exit(1)

if __name__ == "__main__":
    test_ik_integration()
