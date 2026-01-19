
import numpy as np
import numpy.linalg as la
try:
    from ..CharacterData import matrix
    from ..CharacterData import transformations as tm
except ImportError:
    try:
        import CharacterData.matrix as matrix
        import CharacterData.transformations as tm
    except ImportError:
        from .CharacterData import matrix
        from .CharacterData import transformations as tm

class PoseTransfer:
    """
    Handles transferring OpenPose skeleton data to MakeHuman skeleton.
    """
    
    # Mapping derived from find_bones.py
    # OpenPose Key -> MakeHuman Bone Name
    # Note: R/L mirroring (OpenPose Right = MH Left)
    # Mapping derived from game_engine.mhskel
    # OpenPose Key -> MakeHuman Bone Name
    # Note: R/L mirroring (OpenPose Right = MH Left)
    MAPPING = {
        "neck": "neck_01",
        
        # Arm Left (OP) -> Arm Right (MH)
        "l_shoulder": "upperarm_r", 
        "l_elbow": "lowerarm_r",
        "l_wrist": "hand_r", 
        
        # Arm Right (OP) -> Arm Left (MH)
        "r_shoulder": "upperarm_l",
        "r_elbow": "lowerarm_l",
        "r_wrist": "hand_l", 
        
        # Leg Left (OP) -> Leg Right (MH)
        "l_hip": "thigh_r", 
        "l_knee": "calf_r",
        "l_ankle": "foot_r",
        
        # Leg Right (OP) -> Leg Left (MH)
        "r_hip": "thigh_l",
        "r_knee": "calf_l",
        "r_ankle": "foot_l",
    }
    
    # Bone Chains to solve (Parent -> Child -> Target)
    # Each entry: (BoneName, TargetMetric_Start, TargetMetric_End)
    # Using OpenPose joint names as Start/End
    CHAINS = [
        # Right Arm (MH Left)
        ("upperarm_l", "r_shoulder", "r_elbow"),
        ("lowerarm_l", "r_elbow", "r_wrist"),
        
        # Left Arm (MH Right)
        ("upperarm_r", "l_shoulder", "l_elbow"),
        ("lowerarm_r", "l_elbow", "l_wrist"),
        
        # Right Leg (MH Left)
        ("thigh_l", "r_hip", "r_knee"), 
        ("calf_l", "r_knee", "r_ankle"),
        
        # Left Leg (MH Right)
        ("thigh_r", "l_hip", "l_knee"),
        ("calf_r", "l_knee", "l_ankle"),
    ]
    
    def __init__(self, skeleton):
        self.skeleton = skeleton
        # Transform parameters derived from find_bones.py
        # Scale: Matches Debugger (65.0)
        self.scale = 65.0 
        self.offset = np.array([0, -77.3, 0.45], dtype=np.float32)

    def apply_pose(self, openpose_data):
        """
        Apply OpenPose data to the skeleton bones.
        openpose_data: dict of {joint_name: {x,y,z}}
        """
        
        # 0. Reset pose first?
        # self.skeleton.resetPose() # If such method exists, or set matPose to Identity
        for bone in self.skeleton.getBones():
            bone.matPose = np.identity(4, dtype=np.float32)
            bone.update() # Update global matrices
            
        # 1. Convert debug data to numpy dictionary
        op_dict = {}
        for k, v in openpose_data.items():
            # Convert dict to array
            op_dict[k] = np.array([v['x'], v['y'], v['z']], dtype=np.float32)

        # 2. Iterate over chains and apply rotations
        for bone_name, op_start, op_end in self.CHAINS:
            if op_start not in op_dict or op_end not in op_dict:
                continue
                
            bone = self.skeleton.getBone(bone_name)
            if not bone:
                continue
                
            # Get target vector in User Space
            v_target_user = op_dict[op_end] - op_dict[op_start]
            
            # CRITICAL FIX: Invert Y axis.
            # OpenPose/Screen: Y increases DOWN.
            # MakeHuman/World: Y increases UP.
            v_target_user[1] = -v_target_user[1]
            
            # The User Space is scaled/offset relative to simplified MH Space.
            # But Directions (Vectors) are only affected by Scale.
            # v_target_mh = v_target_user / scale
            # Since we normalize later, scale doesn't matter for specific direction.
            # However, we need to map the vector into the Bone's Parent Space to calculate local rotation.
            
            # Current Bone Vector in Global Rest Space
            # bone.headPos/tailPos are in Global Rest Space (MH Decimeters)
            # v_bone_rest = bone.tailPos - bone.headPos
            
            # Wait, apply_pose creates `matPose`.
            # We need to calculate rotation R such that:
            # R * v_bone_rest matches v_target (in direction).
            
            # But the bone might already be rotated by parent!
            # We work in hierarchies.
            # If parent is rotated, bone.matPoseGlobal is updated.
            # We want: GlobalVector(Bone) || TargetVector.
            # GlobalVector(Bone) = (ParentGlobal * LocalRotation * RestVectorDirection) -- roughly.
            
            # Algorithm:
            # 1. Calculate Target Vector in Global MH Space.
            #    We assume the User Space aligns with Global MH Space (just scaled).
            #    So v_target_global = v_target_user (normalized).
            
            target_dir = matrix.normalize(v_target_user)
            
            # 2. Calculate Parent's global rotation.
            #    If bone has parent, parent.matPoseGlobal handles transform up to parent.
            #    We need to transform target_dir into Local Rest Space of the bone.
            #    Inv(ParentGlobal) * target_dir?
            #    Actually, `matRestGlobal` defines the bind pose.
            #    Let's stick to standard recursive IK-like alignment.
            
            # Get current global position of head (after parent updates)
            # We need to ensure parents are updated first. (CHAINS are ordered roughly root-down).
            
            # Current Global Matrix of the Bone (Calculated from hierarchy with current matPose assumed Identity)
            # Actually, we are SETTING matPose.
            # So we take the parent's current MatPoseGlobal.
            
            if bone.parent:
                parent_mat = bone.parent.matPoseGlobal
            else:
                parent_mat = np.identity(4, dtype=np.float32)
                
            # The bone's Rest transform relative to parent is `matRestRelative`.
            # So before applying `matPose`, the bone is at: ParentGlobal * matRestRelative.
            
            pre_pose_mat = np.dot(parent_mat, bone.matRestRelative) 
            
            # Extract rotation from pre_pose_mat to transform the Reference Vector?
            # Simpler: Use points.
            
            # We want to rotate the vector (0, length, 0) (or whatever bone axis is) 
            # to point at Target in Local Space.
            
            # In MH Skeleton, bones point along Y axis is NOT guaranteed. 
            # Usually Y or Z.
            # Let's use `bone.tailPos - bone.headPos` from REST pose to find local axis.
            # `head` and `tail` in Bone object are Global Rest coords.
            
            rest_dir_global = matrix.normalize(bone.tailPos - bone.headPos)
            # Determine Rest Direction
            rest_dir_global = matrix.normalize(np.array(bone.tailPos) - np.array(bone.headPos))
            
            # Transform TargetDir (Global) into Local Space
            inv_pre_pose = la.inv(pre_pose_mat)
            local_target_hom = np.dot(inv_pre_pose, np.array([target_dir[0], target_dir[1], target_dir[2], 0]))
            local_target = matrix.normalize(np.array(local_target_hom).flatten()[:3])
            
            # Find Local Rest Direction
            local_axis_hom = np.dot(inv_pre_pose, np.array([rest_dir_global[0], rest_dir_global[1], rest_dir_global[2], 0]))
            local_axis = matrix.normalize(np.array(local_axis_hom).flatten()[:3])
            
            # Rotation (Alignment)
            dot = np.dot(local_axis, local_target)
            if dot > 0.999: # Already aligned
                rot_mat = np.identity(4, dtype=np.float32)
            elif dot < -0.999: # Opposed
                rot_mat = matrix.rotx(180)
            else:
                axis = np.cross(local_axis, local_target)
                axis = matrix.normalize(axis)
                angle_rad = np.arccos(dot)
                angle_deg = np.degrees(angle_rad)
                rot_mat = matrix.rotate(angle_deg, axis)
                
            # Scaling - DISABLED to prevent explosion
            # target_len_user = matrix.magnitude(v_target_user)
            # target_len_mh = target_len_user / self.scale
            
            scale_y = 1.0
            # if bone.length > 0.001:
            #    scale_y = target_len_mh / bone.length
                
            scale_mat = np.identity(4, dtype=np.float32)
            
            # Apply scale along the BONE AXIS (local_axis)
            # v = local_axis.reshape(3, 1)
            # scale_mat_3 = np.identity(3) + (scale_y - 1.0) * np.dot(v, v.T)
            # scale_mat[:3, :3] = scale_mat_3
            
            # Combine: R * S
            bone.matPose = np.dot(rot_mat, scale_mat)
            
            # Update bone global to propagate to children for next iteration
            bone.update()
