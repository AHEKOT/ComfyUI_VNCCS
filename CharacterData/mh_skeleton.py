
import json
import numpy as np
import numpy.linalg as la
from collections import OrderedDict
import os

from . import matrix
from . import transformations as tm

class VertexBoneWeights(object):
    """
    Weighted vertex to bone assignments.
    """
    def __init__(self, data, vertexCount=None, rootBone="root"):
        self._vertexCount = None
        self._wCounts = None
        self._nWeights = None
        self.rootBone = rootBone
        self._data = self._build_vertex_weights_data(data, vertexCount, rootBone)
        self._calculate_num_weights()
        self._compiled = {}
        self.name = ""

    @staticmethod
    def fromFile(filename, vertexCount=None, rootBone="root"):
        with open(filename, 'r', encoding='utf-8') as f:
            weightsData = json.load(f, object_pairs_hook=OrderedDict)
        result = VertexBoneWeights(weightsData['weights'], vertexCount, rootBone)
        result.name = weightsData.get('name', "")
        return result

    @property
    def data(self):
        return self._data

    def _calculate_num_weights(self):
        self._wCounts = np.zeros(self._vertexCount, dtype=np.uint32)
        for bname, wghts in list(self._data.items()):
            vs, _ = wghts
            self._wCounts[vs] += 1
        self._nWeights = max(self._wCounts) if len(self._wCounts) > 0 else 0

    def _build_vertex_weights_data(self, vertexWeightsDict, vertexCount=None, rootBone="root"):
        WEIGHT_THRESHOLD = 1e-4
        
        # Check if already in internal format
        first_entry = list(vertexWeightsDict.keys())[0] if len(vertexWeightsDict) > 0 else None
        if len(vertexWeightsDict) > 0 and \
           len(vertexWeightsDict[first_entry]) == 2 and \
           isinstance(vertexWeightsDict[first_entry], tuple):
             if vertexCount is not None:
                 self._vertexCount = vertexCount
             else:
                 self._vertexCount = max([vn for vg in list(vertexWeightsDict.values()) for vn in vg[0]])+1
             return vertexWeightsDict
        
        if vertexCount is not None:
            vcount = vertexCount
        else:
             vals = [vn for vg in list(vertexWeightsDict.values()) for vn, _ in vg]
             vcount = max(vals) + 1 if vals else 0
        self._vertexCount = vcount

        wtot = np.zeros(vcount, np.float32)
        for vgroup in list(vertexWeightsDict.values()):
            for item in vgroup:
                vn,w = item
                wtot[vn] += w

        boneWeights = OrderedDict()
        for bname,vgroup in list(vertexWeightsDict.items()):
            if len(vgroup) == 0: continue
            weights = []
            verts = []
            v_lookup = {}
            for vn,w in vgroup:
                if vn in v_lookup:
                    v_idx = v_lookup[vn]
                    weights[v_idx] += w/wtot[vn]
                else:
                    v_lookup[vn] = len(verts)
                    verts.append(vn)
                    weights.append(w/wtot[vn])
            verts = np.asarray(verts, dtype=np.uint32)
            weights = np.asarray(weights, np.float32)
            
            i_s = np.argsort(verts)
            verts = verts[i_s]
            weights = weights[i_s]
            
            i_s = np.argwhere(weights > WEIGHT_THRESHOLD)[:,0]
            verts = verts[i_s]
            weights = weights[i_s]
            boneWeights[bname] = (verts, weights)

        if rootBone not in list(boneWeights.keys()):
            vs = []
            ws = []
        else:
            vs,ws = boneWeights[rootBone]
            vs = list(vs)
            ws = list(ws)
            
        rw_i = np.argwhere(wtot == 0)[:,0]
        if len(rw_i) > 0:
            vs.extend(rw_i)
            ws.extend(np.ones(len(rw_i), dtype=np.float32))
            
        if len(vs) > 0:
            boneWeights[rootBone] = (np.asarray(vs, dtype=np.uint32), np.asarray(ws, dtype=np.float32))

        return boneWeights


def get_normal_from_plane(skel, plane_name, plane_defs, mesh):
    if plane_name not in plane_defs:
        # log.warning
        return np.array([0,0,1], dtype=np.float32)

    joint_names = plane_defs[plane_name]
    j1,j2,j3 = joint_names[:3] # Ensure 3
    
    p1 = skel.getJointPosition(j1, mesh)[:3] 
    p2 = skel.getJointPosition(j2, mesh)[:3]
    p3 = skel.getJointPosition(j3, mesh)[:3]
    
    pvec = matrix.normalize(p2-p1)
    yvec = matrix.normalize(p3-p2) # Note: direction p2->p3? 
    # Logic in MH: yvec = normalize(p3-p2)
    # return normalize(cross(yvec, pvec))
    
    return matrix.normalize(np.cross(yvec, pvec))

def getMatrix(head, tail, normal):
    """
    Generate a bone local rest matrix.
    Copy from makehuman/shared/skeleton.py
    """
    mat = np.identity(4, dtype=np.float32)
    bone_direction = tail - head
    
    # Check for zero length bone
    mag = matrix.magnitude(bone_direction[:3])
    if mag < 1e-6:
        # Zero length bone: Use identity rotation
        mat[:3,3] = head[:3]
        return mat
        
    bone_direction = bone_direction / mag # Already calc mag
    normal = matrix.normalize(normal[:3])

    # We want an orthonormal base
    # Take Z as perpendicular to normal and bone_direction
    z_axis = matrix.normalize(np.cross(normal, bone_direction))
    
    # If z_axis is 0 (normal parallel to bone), pick arbitrary
    if matrix.magnitude(z_axis) < 1e-6:
        z_axis = np.array([0,0,1], dtype=np.float32) 
        if abs(np.dot(z_axis, bone_direction)) > 0.9:
            z_axis = np.array([1,0,0], dtype=np.float32)
        z_axis = matrix.normalize(np.cross(z_axis, bone_direction))

    # Calculate X as orthogonal on Y and Z
    x_axis = matrix.normalize(np.cross(bone_direction, z_axis))

    # Construct orthonormal base
    mat[:3,0] = x_axis[:3]          # bone local X axis
    mat[:3,1] = bone_direction[:3]  # bone local Y axis
    mat[:3,2] = z_axis[:3]          # bone local Z axis

    # Add head position as translation
    mat[:3,3] = head[:3]

    return mat

class Bone(object):
    def __init__(self, skel, name, parentName, headJoint, tailJoint, roll=0, reference_bones=None, weight_reference_bones=None):
        self.name = name
        self.skeleton = skel
        self.headJoint = headJoint
        self.tailJoint = tailJoint
        self.headPos = np.zeros(3,dtype=np.float32)
        self.tailPos = np.zeros(3,dtype=np.float32)
        self.roll = roll
        self.length = 0
        
        self.children = []
        if parentName:
            self.parent = skel.getBone(parentName)
            if self.parent:
                self.parent.children.append(self)
        else:
            self.parent = None
            
        self.level = self.parent.level + 1 if self.parent else 0
        
        self.reference_bones = []
        if reference_bones is not None:
             if not isinstance(reference_bones, list): reference_bones = [reference_bones]
             self.reference_bones.extend(set(reference_bones))
             
        self._weight_reference_bones = None
        if weight_reference_bones is not None:
             if not isinstance(weight_reference_bones, list): weight_reference_bones = [weight_reference_bones]
             self._weight_reference_bones = list(set(weight_reference_bones))
             
        self.matRestGlobal = None
        self.matRestRelative = None
        self.matPose = np.identity(4, np.float32)
        self.matPoseGlobal = None
        self.matPoseVerts = None
        
    def updateJointPositions(self, mesh):
        self.headPos[:] = self.skeleton.getJointPosition(self.headJoint, mesh)
        self.tailPos[:] = self.skeleton.getJointPosition(self.tailJoint, mesh)
        
    def build(self, mesh=None):
        head3 = np.array(self.headPos[:3], dtype=np.float32)
        tail3 = np.array(self.tailPos[:3], dtype=np.float32)
        
        # Calculate normal
        normal = self.get_normal(mesh)
        
        self.matRestGlobal = getMatrix(head3, tail3, normal)
        self.length = matrix.magnitude(tail3 - head3)
        
        if self.parent:
             # Relative to parent
             self.matRestRelative = np.dot(la.inv(self.parent.matRestGlobal), self.matRestGlobal)
        else:
             self.matRestRelative = self.matRestGlobal
             
        self.update()
        
    def update(self):
        # Update pose matrices
        if self.parent:
            self.matPoseGlobal = np.dot(self.parent.matPoseGlobal, np.dot(self.matRestRelative, self.matPose))
        else:
            self.matPoseGlobal = np.dot(self.matRestRelative, self.matPose)
            
        # Transform from bind pose to current pose
        # matPoseVerts = GlobalPose * Inverse(GlobalBind)
        # But GlobalBind is just matRestGlobal
        try:
            invRest = la.inv(self.matRestGlobal)
            self.matPoseVerts = np.dot(self.matPoseGlobal, invRest)
        except la.LinAlgError:
            # Singular matrix - use identity as fallback
            self.matPoseVerts = np.identity(4, np.float32)

    def get_normal(self, mesh=None):
        if self.roll == 0:
            return np.array([0,0,1], dtype=np.float32)
            
        if isinstance(self.roll, list):
            # Average normals
            normal = np.zeros(3, dtype=np.float32)
            for plane_name in self.roll:
                normal += get_normal_from_plane(self.skeleton, plane_name, self.skeleton.planes, mesh)
            return matrix.normalize(normal)
        else:
            # Single plane
            return get_normal_from_plane(self.skeleton, self.roll, self.skeleton.planes, mesh)


class Skeleton(object):
    def __init__(self, name="Skeleton"):
        self.name = name
        self.bones = {}
        self.boneslist = []
        self.roots = []
        self.joint_pos_idxs = {}
        self.planes = {}
        self.vertexWeights = None
        self.scale = 1.0
        
    def fromFile(self, filepath, mesh=None):
        with open(filepath, 'r', encoding='utf-8') as f:
            skelData = json.load(f, object_pairs_hook=OrderedDict)
            
        self.name = skelData.get("name", self.name)
        
        joints = skelData.get("joints", {})
        for joint_name, v_idxs in joints.items():
            if isinstance(v_idxs, list) and len(v_idxs) > 0:
                self.joint_pos_idxs[joint_name] = v_idxs
                
        self.planes = skelData.get("planes", {})
        
        # Breadth-first sort bones
        input_bones = skelData["bones"]
        breadthfirst_bones = []
        
        # Naive sort
        pending = list(input_bones.keys())
        added = set()
        
        while pending:
            progress = False
            for bname in pending[:]:
                bdef = input_bones[bname]
                parent = bdef.get("parent", None)
                if not parent or parent in added:
                    breadthfirst_bones.append(bname)
                    added.add(bname)
                    pending.remove(bname)
                    progress = True
            if not progress and pending:
                print(f"Warning: Circular or missing parent dependency for bones: {pending}")
                break
                
        for bone_name in breadthfirst_bones:
            bone_defs = input_bones[bone_name]
            rotation_plane = bone_defs.get("rotation_plane", 0)
            if rotation_plane == [None, None, None]: rotation_plane = 0
            
            self.addBone(bone_name, bone_defs.get("parent", None), 
                         bone_defs["head"], bone_defs["tail"], 
                         rotation_plane, 
                         bone_defs.get("reference", None), 
                         bone_defs.get("weights_reference", None))
                         
        if mesh:
            self.updateJointPositions(mesh)
            
        if "weights_file" in skelData and skelData["weights_file"]:
             weights_file = skelData["weights_file"]
             # Resolve relative to mhskel file
             w_path = os.path.join(os.path.dirname(filepath), weights_file)
             if os.path.exists(w_path):
                 count = len(mesh.vertices) if mesh else None
                 self.vertexWeights = VertexBoneWeights.fromFile(w_path, count, self.roots[0].name)
             else:
                 print(f"Weights file not found: {w_path}")
        else:
             # Fallback: Try default_weights.mhw
             w_path = os.path.join(os.path.dirname(filepath), "default_weights.mhw")
             if os.path.exists(w_path):
                 count = len(mesh.vertices) if mesh else None
                 print(f"[Skeleton] Loading fallback weights from {w_path}")
                 self.vertexWeights = VertexBoneWeights.fromFile(w_path, count, self.roots[0].name)
                 
        # Retarget weights if referencing is used (e.g. Game Engine config)
        if self.vertexWeights:
            self._retarget_weights()

    def _retarget_weights(self):
        """
        Retarget weights from referenced bones (e.g. upperarm01+upperarm02) 
        to the actual bone (upperarm_l), if 'weights_reference' structure is present.
        """
        # We work directly on self.vertexWeights.data (OrderedDict)
        # Create a new dictionary for the retargeted weights
        new_weights_data = OrderedDict()
        
        # Get source bone names available in loaded weights
        source_bones = set(self.vertexWeights.data.keys())
        
        has_retargeting = False
        
        for bone in self.boneslist:
            bname = bone.name
            
            # Use explicit weight reference if available, otherwise fallback to standard reference
            refs = bone._weight_reference_bones
            if not refs and bone.reference_bones:
                refs = bone.reference_bones
            
            # If bone references other bones for weights (or position)
            if refs:
                has_retargeting = True
                
                # Collect weights from all referenced bones
                combined_verts = {} # vert_idx -> weight
                
                for ref_name in refs:
                    if ref_name in self.vertexWeights.data:
                        vs, ws = self.vertexWeights.data[ref_name]
                        for i, v in enumerate(vs):
                            if v not in combined_verts:
                                combined_verts[v] = 0.0
                            combined_verts[v] += ws[i]
                            
                # Reconstruct arrays
                if combined_verts:
                    vs = np.array(list(combined_verts.keys()), dtype=np.uint32)
                    ws = np.array(list(combined_verts.values()), dtype=np.float32)
                    
                    # Sort by vertex index
                    idx_sorted = np.argsort(vs)
                    vs = vs[idx_sorted]
                    ws = ws[idx_sorted]
                    
                    new_weights_data[bname] = (vs, ws)
            
            # If no reference, check if bone name matches source directly
            elif bname in source_bones:
                new_weights_data[bname] = self.vertexWeights.data[bname]
                
        if has_retargeting:
            # 2. ADDITIONAL CLEANUP: Map orphaned weight bones (muscles/helpers) to game_engine bones
            # GameEngine skeleton typically ignores substantial helper bones found in Default.
            # We must map them to prevent mesh collapsing (vertices with 0 weight).
            extra_mapping = {
                "clavicle_l": ["scapula.L", "trapezius.L"],
                "clavicle_r": ["scapula.R", "trapezius.R"],
                "upperarm_l": ["deltoid.L"],
                "upperarm_r": ["deltoid.R"],
                "spine_03": ["latissimus.L", "latissimus.R", "pectoralis.L", "pectoralis.R"],
                "pelvis": ["gluteus.L", "gluteus.R"],
                # Map major face bones to head to prevent face collapse
                "head": [
                    "jaw", "eye.L", "eye.R", 
                    "tongue01", "tongue02", "tongue03", "tongue04",
                    "upperlid.L", "upperlid.R", "lowerlid.L", "lowerlid.R"
                ]
            }
            
            for target_bone, sources in extra_mapping.items():
                if target_bone in self.vertexWeights.data:
                    # Get existing target weights
                    t_vs, t_ws = new_weights_data.get(target_bone, (np.array([], dtype=np.uint32), np.array([], dtype=np.float32)))
                    
                    # Convert to dict for merging
                    combined = dict(zip(t_vs, t_ws))
                    
                    added = False
                    for src in sources:
                        if src in self.vertexWeights.data:
                            s_vs, s_ws = self.vertexWeights.data[src]
                            for i, v in enumerate(s_vs):
                                if v not in combined:
                                    combined[v] = 0.0
                                combined[v] += s_ws[i]
                            added = True
                            
                    if added:
                        # Reconstruct arrays
                        vs = np.array(list(combined.keys()), dtype=np.uint32)
                        ws = np.array(list(combined.values()), dtype=np.float32)
                        idx_sorted = np.argsort(vs)
                        new_weights_data[target_bone] = (vs[idx_sorted], ws[idx_sorted])
            
            print(f"[Skeleton] Retargeted weights for {len(new_weights_data)} bones (including cleanup).")
            # Replace data
            self.vertexWeights._data = new_weights_data
            # Re-calculate metadata
            self.vertexWeights._calculate_num_weights()

    def addBone(self, name, parentName, head, tail, roll, ref=None, w_ref=None):
        bone = Bone(self, name, parentName, head, tail, roll, ref, w_ref)
        self.bones[name] = bone
        self.boneslist.append(bone)
        if not bone.parent:
            self.roots.append(bone)
            
    def getBone(self, name):
        return self.bones.get(name)
        
    def getBones(self):
        return self.boneslist

    def getJointPosition(self, joint_name, mesh):
        if joint_name in self.joint_pos_idxs:
            v_idxs = self.joint_pos_idxs[joint_name]
            # Assumes mesh.vertices is Numpy array (N,3)
            verts = mesh.vertices[v_idxs]
            return verts.mean(axis=0)
        else:
            # Fallback / Placeholder
            # In MH this scans facegroups. We probably don't need it if skel is good.
            return np.array([0,0,0], dtype=np.float32)

    def updateJointPositions(self, mesh):
        for bone in self.boneslist:
            bone.updateJointPositions(mesh)
            bone.build(mesh)
            
    def copy(self):
        # Create new empty skeleton
        new_skel = Skeleton(self.name)
        new_skel.joints_pos_idxs = self.joint_pos_idxs # Ref copy OK, read only
        new_skel.planes = self.planes # Ref copy OK
        new_skel.vertexWeights = self.vertexWeights # Ref copy OK
        
        # Copy bones
        # We must maintain hierarchy
        # self.boneslist is breadth-first, so parents come before children
        for bone in self.boneslist:
            parent_name = bone.parent.name if bone.parent else None
            new_skel.addBone(bone.name, parent_name, 
                             bone.headJoint, bone.tailJoint, 
                             bone.roll, 
                             bone.reference_bones, 
                             bone._weight_reference_bones)
                             
        new_skel.scale = self.scale
        return new_skel




    # IMPORTANT: Need proper get_normal implementation for Bone.build to work correctly 
    # Otherwise skeletons will come out twisted.
    # Re-implementing logic from MH Bone.get_normal inside Bone class above would be better.
