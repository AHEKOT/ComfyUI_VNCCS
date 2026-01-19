
import sys
import os
import json
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from CharacterData.mh_skeleton import Skeleton
from CharacterData.obj_loader import load_obj

def find_closest_bones():
    # 1. Load Skeleton and Mesh
    char_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "CharacterData"))
    mh_path = os.path.join(char_data_path, "makehuman")
    
    # Paths
    base_obj_path = os.path.join(mh_path, "data", "3dobjs", "base.obj")
    if not os.path.exists(base_obj_path):
        base_obj_path = os.path.join(mh_path, "makehuman", "data", "3dobjs", "base.obj")
        
    skel_path = os.path.join(mh_path, "makehuman", "data", "rigs", "default.mhskel")
    
    print(f"Loading mesh from {base_obj_path}")
    mesh = load_obj(base_obj_path)
    
    print(f"Loading skeleton from {skel_path}")
    skel = Skeleton()
    skel.fromFile(skel_path, mesh)
    
    # 2. User Debug Data (OpenPose Joint -> 3D Coordinate)
    debug_data = {
      "neck": { "x": 0, "y": 273.7, "z": 4.5 },
      "nose": { "x": 0, "y": 323.5, "z": 4.5 },
      "r_shoulder": { "x": 64, "y": 194.9, "z": 4.5 },
      "r_elbow": { "x": 150.5, "y": 110.7, "z": 11.4 },
      "r_wrist": { "x": 239.7, "y": 18.7, "z": 100.8 },
      "l_shoulder": { "x": -64, "y": 201.4, "z": 13.6 },
      "l_elbow": { "x": -157.3, "y": 98.9, "z": 12.7 },
      "l_wrist": { "x": -245.6, "y": 12.8, "z": 108.8 },
      "r_hip": { "x": 59.4, "y": -107.7, "z": 4.5 },
      "r_knee": { "x": 79.6, "y": -273.1, "z": 9.8 },
      "r_ankle": { "x": 98.6, "y": -521.3, "z": 1.2 },
      "l_hip": { "x": -60.5, "y": -109.9, "z": 4.5 },
      "l_knee": { "x": -77.9, "y": -269.7, "z": 19.9 },
      "l_ankle": { "x": -101.6, "y": -517.8, "z": 12.9 },
      "r_eye": { "x": 14, "y": 338.5, "z": 4.5 },
      "l_eye": { "x": -14, "y": 338.5, "z": 4.5 },
      "r_ear": { "x": 29, "y": 328.5, "z": 4.5 },
      "l_ear": { "x": -29, "y": 328.5, "z": 4.5 }
    }
    
    # 3. Find closest bones
    # Debug Bounds
    verts = mesh.vertices
    print(f"Mesh Vertices Range: X[{verts[:,0].min():.2f}, {verts[:,0].max():.2f}] Y[{verts[:,1].min():.2f}, {verts[:,1].max():.2f}] Z[{verts[:,2].min():.2f}, {verts[:,2].max():.2f}]")
    center = verts.mean(axis=0)
    print(f"Mesh Center: {center}")

    # List all bones
    print("\n--- All Skeleton Bones (Sorted by Y height) ---")
    bones_by_y = sorted(skel.getBones(), key=lambda b: b.headPos[1], reverse=True)
    for b in bones_by_y:
        print(f"{b.name:<20} Pos: {b.headPos}")

    # Try to find transform: User = Bone * S + T
    # Heuristic: Match 'neck' debug point to 'neck' bone (if exists) or 'spine03'
    # And 'ankle' to 'foot'
    
    # Let's just find closest using raw mesh space first (assuming user mesh was raw)
    # If user mesh was at 0,0,0 and Scale 65.
    # And original mesh is Y[0..17]
    # Then User Y = Bone Y * 65.
    
    # Let's try to find scale using neck and ankle.
    # Debug: Neck Y=273. Ankle Y=-520. Delta=793.
    # Mesh: Find 'neck' and 'ankle' bones.
    
    neck_bone = skel.getBone("neck01") or skel.getBone("neck") or skel.getBone("head")
    ankle_bone = skel.getBone("lowerleg01.R") # ankle is tail of lowerleg? or head of foot?
    # Let's look for "foot.R"
    foot_bone = skel.getBone("foot.R")
    
    if neck_bone and foot_bone:
        mh_h = neck_bone.headPos[1] - foot_bone.headPos[1]
        user_h = 273.7 - (-521.2) # using R_ankle
        scale = user_h / mh_h
        print(f"\nEstimated Scale: {scale:.2f} (MH Delta: {mh_h:.2f}, User Delta: {user_h:.2f})")
        
        # Estimate offset
        # User = Bone * Scale + Offset
        # Offset = User - Bone * Scale
        offset = np.array([0, 273.7, 4.5]) - neck_bone.headPos * scale
        print(f"Estimated Offset: {offset}")
    
        # Apply transform and match
        print("\n--- Mapping with Estimated Transform ---")
        bone_positions = {}
        for bone in skel.getBones():
            bone_positions[bone.name] = bone.headPos * scale + offset

        mapping = {}
        used_bones = set()
        
        for op_name, coords in debug_data.items():
            target = np.array([coords['x'], coords['y'], coords['z']])
            
            # Simple greedy matching logic?
            # Prefer bones that sound like the target?
            
            min_dist = float('inf')
            closest_bone = None
            
            for b_name, b_pos in bone_positions.items():
                dist = np.linalg.norm(target - b_pos)
                if dist < min_dist:
                    min_dist = dist
                    closest_bone = b_name
            
            print(f"  {op_name:<12} -> {closest_bone:<20} (dist: {min_dist:.2f})")
            mapping[op_name] = closest_bone
    else:
        print("Could not find reference bones for scaling.")


if __name__ == "__main__":
    find_closest_bones()
