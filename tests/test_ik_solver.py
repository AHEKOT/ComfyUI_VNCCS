
import sys
import os
import math
import numpy as np
import numpy.linalg as la

# --- SELF-CONTAINED MOCKS ---
class matrix:
    @staticmethod
    def translate(t):
        m = np.identity(4)
        m[:3,3] = t
        return m

class transformations:
    @staticmethod
    def rotation_matrix(angle, axis):
        axis = np.array(axis)
        n = np.linalg.norm(axis)
        if n > 0: axis /= n
        a = math.cos(angle / 2.0)
        b, c, d = axis * math.sin(angle / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array([
            [aa + bb - cc - dd, 2 * (bc - ad), 2 * (bd + ac), 0],
            [2 * (bc + ad), aa + cc - bb - dd, 2 * (cd - ab), 0],
            [2 * (bd - ac), 2 * (cd + ab), aa + dd - bb - cc, 0],
            [0, 0, 0, 1]
        ])
    @staticmethod
    def euler_from_matrix(matrix, axes='sxyz'):
        return 0,0,0

# --- INLINE IKSOLVER (Copy-Paste from nodes/ik_solver.py because exec is fragile) ---
# I will copy paste the class definition here to ensure 100% correct execution environment.
class IKSolver:
    """
    Simple CCD (Cyclic Coordinate Descent) IK Solver for MakeHuman skeletons.
    """
    
    @staticmethod
    def solve_ik(skeleton, bone_name, target_pos, chain_length=3, max_iterations=10, tolerance=0.01):
        """
        Solve IK for a specific bone to reach target_pos.
        """
        
        # 1. Build Chain
        chain = []
        current = skeleton.getBone(bone_name)
        if not current:
            print(f"IKSolver: Bone {bone_name} not found.")
            return {}
            
        for _ in range(chain_length):
            chain.append(current)
            if not current.parent:
                break
            current = current.parent
            
        # chain[0] is End Effector, chain[-1] is Root of chain
        target = np.array(target_pos)
        
            # 2. CCD Loop
        for iteration in range(max_iterations):
            end_effector = chain[0]
            current_pos_vec = np.array(end_effector.headPos) 
            
            # Start of loop: Update all
            skeleton.update() 
            
            # Calculate global visual pos of End Effector Head
            mat_skin = end_effector.matPoseVerts
            h_vec = np.array([end_effector.headPos[0], end_effector.headPos[1], end_effector.headPos[2], 1.0])
            cur_global = np.dot(mat_skin, h_vec)[:3]
            
            dist = la.norm(target - cur_global)
            if dist < tolerance:
                break
                
            # Iterate chain from Tip to Root
            for bone in chain:
                # 1. Get Pivot (Bone Head) in Global Space
                mat_skin_b = bone.matPoseVerts
                b_head_rest = np.array([bone.headPos[0], bone.headPos[1], bone.headPos[2], 1.0])
                pivot = np.dot(mat_skin_b, b_head_rest)[:3]
                
                # 2. Vector from Pivot to Effector
                to_effector = cur_global - pivot
                
                # 3. Vector from Pivot to Target
                to_target = target - pivot
                
                # Normalize
                len_eff = la.norm(to_effector)
                len_tgt = la.norm(to_target)
                
                if len_eff < 1e-6 or len_tgt < 1e-6:
                    continue
                    
                to_effector /= len_eff
                to_target /= len_tgt
                
                # 4. Rotation needed (Quat/Axis-Angle)
                axis = np.cross(to_effector, to_target)
                sin_angle = la.norm(axis)
                cos_angle = np.dot(to_effector, to_target)
                
                if sin_angle < 1e-6:
                    continue
                    
                axis /= sin_angle
                angle = np.arctan2(sin_angle, cos_angle)
                
                if bone.name == 'mid':
                    print(f"Iter {iteration} Bone {bone.name}: Pivot={pivot} Eff={cur_global} Tgt={target}")
                    print(f"  to_eff={to_effector} to_tgt={to_target}")
                    print(f"  Axis={axis} Angle={math.degrees(angle)}")

                
                # 5. Apply to Bone Local Rotation
                tm = transformations # Use local mock
                rot_delta_global = tm.rotation_matrix(angle, axis) # 4x4
                
                inv_global = la.inv(bone.matPoseGlobal)
                local_axis_v4 = np.dot(inv_global, np.append(axis, 0))
                local_axis = local_axis_v4[:3]
                local_axis /= la.norm(local_axis)
                
                # Create Local Rotation Matrix
                rot_local = tm.rotation_matrix(angle, local_axis)
                
                # Update bone.matPose
                bone.matPose = np.dot(bone.matPose, rot_local)
                
                # Update Skeleton immediately
                skeleton.update()
                
                # Update cur_global for next step loop
                mat_skin_eff = end_effector.matPoseVerts
                cur_global = np.dot(mat_skin_eff, h_vec)[:3]

        # 3. Extract Resulting Eulers
        result = {}
        for bone in chain:
             tm = transformations
             rx, ry, rz = tm.euler_from_matrix(bone.matPose, 'rzyx')
             result[bone.name] = [np.degrees(rx), np.degrees(ry), np.degrees(rz)]
            
        return result

# --- MOCK BONE / SKELETON ---
class MockBone:
    def __init__(self, name, head, tail, parent=None):
        self.name = name
        self.headPos = np.array(head, dtype=float)
        self.tailPos = np.array(tail, dtype=float)
        self.parent = parent
        self.matPose = np.identity(4)
        self.matPoseGlobal = np.identity(4)
        self.matPoseVerts = np.identity(4)
        self.matRestGlobal = np.identity(4); self.matRestGlobal[:3,3] = self.headPos

    def update(self):
        if self.parent:
             self.matPoseGlobal = np.dot(self.parent.matPoseGlobal, self.matPose)
        else:
             self.matPoseGlobal = self.matPose
        self.matPoseVerts = np.dot(self.matPoseGlobal, np.linalg.inv(self.matRestGlobal))

class MockSkeleton:
    def __init__(self):
        self.bones = {}
    def getBone(self, name):
        return self.bones.get(name)
    def update(self):
        # Update Order: Root -> Mid -> Tip
        if 'root' in self.bones: self.bones['root'].update()
        if 'mid' in self.bones: self.bones['mid'].update()
        if 'tip' in self.bones: self.bones['tip'].update()

def test_ik_convergence():
    print("--- Setting up Mock Skeleton ---")
    skel = MockSkeleton()
    
    # Root(0,0,0)->(0,10,0)
    root = MockBone("root", [0,0,0], [0,10,0], None)
    # Mid(0,10,0)->(0,20,0)
    mid = MockBone("mid", [0,10,0], [0,20,0], root)
    # Tip(0,20,0)->(0,21,0)
    tip = MockBone("tip", [0,20,0], [0,21,0], mid)
    
    # Set Local Translation (Rest) via matPose initial state
    root.matPose[:3,3] = [0,0,0]
    mid.matPose[:3,3] = [0,10,0]
    tip.matPose[:3,3] = [0,10,0] 
    
    skel.bones = {'root': root, 'mid': mid, 'tip': tip}
    skel.update()
    
    head_world_rest = np.array([0,20,0,1])
    cur_pos = np.dot(tip.matPoseVerts, head_world_rest)[:3]
    print(f"Initial Tip Pos: {cur_pos}")
    assert np.allclose(cur_pos, [0,20,0], atol=0.1)
    
    # Target: (10, 10, 0)
    target = np.array([10.0, 10.0, 0.0])
    
    print("--- Running Solver ---")
    
    try:
        rotations = IKSolver.solve_ik(skel, 'tip', target, chain_length=2, max_iterations=20)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Solver crashed: {e}")
        exit(1)
        
    skel.update()
    final_pos = np.dot(tip.matPoseVerts, head_world_rest)[:3]
    print(f"Final Tip Pos: {final_pos}")
    
    dist = np.linalg.norm(final_pos - target)
    print(f"Distance: {dist}")
    
    if dist < 1.0:
        print("SUCCESS")
    else:
        print("FAILURE: Distance too high")
        exit(1)

if __name__ == "__main__":
    test_ik_convergence()
