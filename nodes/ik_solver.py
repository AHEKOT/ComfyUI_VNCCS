import numpy as np
import numpy.linalg as la
try:
    from ..CharacterData import matrix
except ImportError:
    try:
        import CharacterData.matrix as matrix
    except ImportError:
        from .CharacterData import matrix

try:
    from ..CharacterData import transformations as tm
except ImportError:
    try:
        import CharacterData.transformations as tm
    except ImportError:
        try:
            from .CharacterData import transformations as tm
        except ImportError:
            import transformations as tm

import math

class IKSolver:
    """
    Simple CCD (Cyclic Coordinate Descent) IK Solver for MakeHuman skeletons.
    """
    
    @staticmethod
    def solve_ik(skeleton, bone_name, target_pos, chain_length=3, max_iterations=10, tolerance=0.01):
        """
        Solve IK for a specific bone to reach target_pos.
        
        Args:
            skeleton: The Skeleton instance.
            bone_name: Name of the end-effector bone (the one dragging).
            target_pos: [x, y, z] target position in World Space.
            chain_length: Number of bones up the hierarchy to modify (e.g. Hand -> LowerArm -> UpperArm = 3).
            max_iterations: CCD iterations.
            tolerance: Distance threshold to stop.
            
        Returns:
            dict: {bone_name: [rx, ry, rz]} of modified rotations (Euler degrees).
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
            # Check distance
            end_effector = chain[0]
            # Use Visual Position (Mesh Space) logic? 
            # Ideally we iterate using current matPoseGlobal state.
            # But wait, matPoseGlobal is updated via bone.update().
            
            # Since we modify matPose, we must re-update the skeleton specific bones?
            # Or just update global matrices.
            # skeleton.update() is expensive if full. But needed.
            
            current_pos = end_effector.headPos # Actually we want TAIL usually for end effector?
            # Or HEAD? Interactive control usually grabs the Joint (Head).
            # If I grab "Hand", I am grabbing the Wrist (Head of Hand).
            current_pos_vec = np.array(end_effector.headPos) # This is REST pos.
            
            # We need GLOBAL CURRENT pos.
            # Start of loop: Update all
            # Start of loop: Update all
            for b in skeleton.getBones():
                b.update() 
            
            # Calculate global visual pos of End Effector Head
            # M_skin * RestPos
            mat_skin = end_effector.matPoseVerts
            h_vec = np.array([end_effector.headPos[0], end_effector.headPos[1], end_effector.headPos[2], 1.0])
            cur_global = np.asarray(np.dot(mat_skin, h_vec)).ravel()[:3]
            
            dist = la.norm(target - cur_global)
            if dist < tolerance:
                break
                
            # Iterate chain from Tip to Root
            for bone in chain:
                # 1. Get Pivot (Bone Head) in Global Space
                # Same visual logic
                mat_skin_b = bone.matPoseVerts
                b_head_rest = np.array([bone.headPos[0], bone.headPos[1], bone.headPos[2], 1.0])
                pivot = np.asarray(np.dot(mat_skin_b, b_head_rest)).ravel()[:3]
                
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
                # Rotate to_effector ALIGN to_target
                
                # Cross product for axis
                axis = np.cross(to_effector, to_target)
                sin_angle = la.norm(axis)
                cos_angle = np.dot(to_effector, to_target)
                
                if sin_angle < 1e-6:
                    continue
                    
                axis /= sin_angle
                angle = np.arctan2(sin_angle, cos_angle)
                
                # Limit angle per step for stability?
                # angle = max(-0.1, min(0.1, angle)) 
                
                # 5. Apply to Bone Local Rotation
                # We need to rotate the bone in its LOCAL space.
                # Current Global Rotation of bone:
                # We want to Apply Global Rotation (Axis, Angle).
                
                # Quat Global Delta
                
                # Create rotation matrix from axis-angle
                rot_delta_global = tm.rotation_matrix(angle, axis) # 4x4
                
                # Convert this Global Rotation to Local Bone Space
                # M_global_new = Rot_delta * M_global_old
                # M_local = (M_parent_global * M_rest_local)^-1 * M_global_new
                
                # Easier approach:
                # bone.matPose is the local rotation (relative to parent/rest).
                # We want to tweak it.
                # Transform Axis to Local Space?
                
                # Get Inverse Global Matrix of the Bone (Rotation only)
                # Actually, bone.matPoseGlobal includes pure rotation? No, it has translation.
                
                # Let's look at how setPose works in skeleton.py setPose:
                # bone.matPose = invRest * GlobalPose * RestGlobal
                
                # We have Rot_delta (Global).
                # New Global Matrix for this bone would be:
                # M_new = Rot_delta * bone.matPoseGlobal
                
                # Now extract new Local Pose from M_new
                # bone.matPose = inv(M_parent_global) * M_new * inv(M_rest_local) ???
                # Wait, bone.matPose is defined such that:
                # M_global = M_parent_global * M_rest_local * bone.matPose --- No.
                
                # R_global = R_parent_global * R_local_rest * R_pose
                # So R_pose = (R_parent_global * R_local_rest)^-1 * R_global
                
                # Let's transform the Global Rotation Delta into Local Space Delta
                # Axis_local = (R_parent_global * R_local_rest)^-1 * Axis_global
                
                # Parent Global
                if bone.parent:
                    m_parent = bone.parent.matPoseGlobal
                else:
                    m_parent = np.identity(4)
                
                # Rest Local (matRestRelative?)
                # Actually simpler:
                # M_global = M_parent * M_pose * M_rest_relative ??? No
                
                # From skeleton.py:
                # M_global = M_parent * M_rest_relative * M_pose  <-- If M_pose is applied at joint
                
                # Let's assume standard MH:
                # matPose is the user rotation. 
                # Let's try to Map Axis to Bone Space.
                
                # Inverse of everything before this bone's rotation:
                # Pre = M_parent * M_rest_relative (or however rest is handled)
                
                # Let's use a simpler heuristic which is often used in simple IK:
                # Transform Axis by Inverse(M_global). (To put it in local frame)
                # But M_global includes the rotation we want to change.
                
                # Transform to bone local
                inv_global = la.inv(bone.matPoseGlobal)
                local_axis_v4 = np.asarray(np.dot(inv_global, np.append(axis, 0))).ravel()
                local_axis = local_axis_v4[:3]
                local_axis /= la.norm(local_axis)
                
                # Create Local Rotation Matrix
                rot_local = tm.rotation_matrix(angle, local_axis)
                
                # Save original translation to prevent drift/tearing
                orig_pos = bone.matPose[:3, 3].copy()
                
                # Update bone.matPose
                bone.matPose = np.dot(bone.matPose, rot_local)
                
                # Restore translation
                bone.matPose[:3, 3] = orig_pos
                
                # Normalize rotation part to prevent scale drift
                # (Gram-Schmidt or just normalize basis vectors)
                # Simple X/Y/Z normalization
                bone.matPose[:3, 0] /= la.norm(bone.matPose[:3, 0])
                bone.matPose[:3, 1] /= la.norm(bone.matPose[:3, 1])
                bone.matPose[:3, 2] /= la.norm(bone.matPose[:3, 2])
                # Re-orthogonalize (optional but good for stability)
                 # X = Y cross Z
                 # Y = Z cross X
                 # Z = X cross Y
                 # For now, simple normalization is usually enough for CCD.
                
                # Helper: Normalize matPose (remove scale/drift)
                # ...
                
                # Update Skeleton immediately to propagate changes for next bone in loop
                for b in skeleton.getBones():
                    b.update()
                
                # Update cur_global for next step loop?
                # Need to recalc end effector pos
                mat_skin_eff = end_effector.matPoseVerts
                cur_global = np.asarray(np.dot(mat_skin_eff, h_vec)).ravel()[:3]

        # 3. Extract Resulting Eulers
        result = {}
        for bone in chain:
            # decomposition to Euler ZYX
            # bone.matPose is 4x4
            pass # We need a helper to Convert matPose rotational part to Euler
            
            # Using transformations.py
            rx, ry, rz = tm.euler_from_matrix(bone.matPose, 'rzyx')
            result[bone.name] = [np.degrees(rx), np.degrees(ry), np.degrees(rz)]
            
        return result
