"""VNCCS - Visual Novel Character Creator Suite for ComfyUI."""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']


__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

WEB_DIRECTORY = "web"

import os, json, inspect
import traceback
def _vnccs_register_endpoint():  # lazy registration to avoid import errors in analysis tools
    try:
        from server import PromptServer
        from aiohttp import web
    except Exception:
        return

    @PromptServer.instance.routes.get("/vnccs/config")
    async def vnccs_get_config(request):
        name = request.rel_url.query.get("name")
        if not name:
            return web.json_response({"error": "name required"}, status=400)
        try:
            from .nodes.character_creator import CharacterCreator
            base = CharacterCreator().base_path
        except Exception:
            base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output", "VN_CharacterCreatorSuit"))
        cfg_path = os.path.join(base, name, f"{name}_config.json")
        if not os.path.exists(cfg_path):
            return web.json_response({"error": "not found", "path": cfg_path}, status=404)
        try:
            with open(cfg_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return web.json_response(data)
        except Exception as e:
            return web.json_response({"error": "read failed", "detail": str(e)}, status=500)

    @PromptServer.instance.routes.get("/vnccs/create")
    async def vnccs_create_character(request):
        name = request.rel_url.query.get("name", "").strip()
        if not name:
            return web.json_response({"error": "name required"}, status=400)
        forbidden = set('/\\:')
        if any(c in forbidden for c in name):
            return web.json_response({"error": "invalid characters"}, status=400)
        defaults = dict(
            existing_character=name,
            background_color="green",
            aesthetics="masterpiece",
            nsfw=False,
            sex="female",
            age=18,
            race="human",
            eyes="blue eyes",
            hair="black long",
            face="freckles",
            body="medium breasts",
            skin_color="",
            additional_details="",
            seed=0,
            negative_prompt="bad quality,worst quality,worst detail,sketch,censor, missing arm, missing leg, distorted body",
            lora_prompt="",
            new_character_name=name,
        )
        try:
            from .nodes.character_creator import CharacterCreator
            cc = CharacterCreator()
            os.makedirs(cc.base_path, exist_ok=True)
            base_char_dir = os.path.join(cc.base_path, name)
            config_path = os.path.join(base_char_dir, f"{name}_config.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                except Exception:
                    existing_data = None
                return web.json_response({
                    "ok": True,
                    "name": name,
                    "existing": True,
                    "config_path": config_path,
                    "data": existing_data,
                })
            # Backward compatibility: drop force_new if method doesn't accept it
            try:
                sig = inspect.signature(cc.create_character)
                if 'force_new' not in sig.parameters and 'force_new' in defaults:
                    defaults.pop('force_new')
            except Exception:
                defaults.pop('force_new', None)
            positive_prompt, seed, negative_prompt, age_lora_strength, sheets_path, faces_path, face_details = cc.create_character(**defaults)
            return web.json_response({
                "ok": True,
                "name": name,
                "seed": seed,
                "sheets_path": sheets_path,
                "faces_path": faces_path,
                "positive_prompt": positive_prompt,
                "negative_prompt": negative_prompt,
                "age_lora_strength": age_lora_strength,
                "face_details": face_details,
                "config_path": os.path.join(base_char_dir, f"{name}_config.json"),
            })
        except Exception as e:
            return web.json_response({
                "error": "create failed",
                "detail": str(e),
                "type": type(e).__name__,
                "trace": traceback.format_exc(),
            }, status=500)

    @PromptServer.instance.routes.get("/vnccs/create_costume")
    async def vnccs_create_costume(request):
        character_name = request.rel_url.query.get("character", "").strip()
        costume_name = request.rel_url.query.get("costume", "").strip()
        if not character_name or not costume_name:
            return web.json_response({"error": "character and costume required"}, status=400)
        forbidden = set('/\\:')
        if any(c in forbidden for c in character_name) or any(c in forbidden for c in costume_name):
            return web.json_response({"error": "invalid characters"}, status=400)
        try:
            from .utils import load_config, save_config, ensure_costume_structure
            config = load_config(character_name)
            if not config:
                config = {"character_info": {}, "costumes": {}}
            if "costumes" not in config:
                config["costumes"] = {}
            if costume_name in config["costumes"]:
                return web.json_response({"error": "Costume already exists"})
            config["costumes"][costume_name] = {
                "face": "",
                "head": "",
                "top": "",
                "bottom": "",
                "shoes": "",
                "negative_prompt": ""
            }
            if save_config(character_name, config):
                ensure_costume_structure(character_name, costume_name)
                return web.json_response({"ok": True, "costume": costume_name})
            else:
                return web.json_response({"error": "Failed to save"}, status=500)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    @PromptServer.instance.routes.get("/vnccs/models/{filename}")
    async def vnccs_get_model(request):
        """Serve FBX model files for 3D pose editor"""
        filename = request.match_info.get("filename", "")
        if not filename.endswith(".fbx"):
            return web.Response(text="Only FBX files allowed", status=400)
        
        # Get the models directory
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        file_path = os.path.join(models_dir, filename)
        
        # Security check - ensure file is within models directory
        if not os.path.abspath(file_path).startswith(os.path.abspath(models_dir)):
            return web.Response(text="Invalid path", status=400)
        
        if not os.path.exists(file_path):
            return web.Response(text=f"Model not found: {filename}", status=404)
        
        try:
            with open(file_path, 'rb') as f:
                return web.Response(
                    body=f.read(),
                    content_type='application/octet-stream',
                    headers={
                        'Content-Disposition': f'inline; filename="{filename}"',
                        'Access-Control-Allow-Origin': '*'
                    }
                )
        except Exception as e:
            return web.Response(text=f"Error reading file: {str(e)}", status=500)
    
    @PromptServer.instance.routes.get("/vnccs/pose_presets")
    async def vnccs_get_pose_presets(request):
        """Get list of available pose presets"""
        try:
            presets_dir = os.path.join(os.path.dirname(__file__), "presets", "poses")
            presets = []
            
            if os.path.exists(presets_dir):
                for filename in sorted(os.listdir(presets_dir)):
                    if filename.endswith('.json'):
                        # Create preset entry
                        preset_id = filename[:-5]  # Remove .json
                        label = preset_id.replace('_', ' ').title()
                        presets.append({
                            "id": preset_id,
                            "label": label,
                            "file": filename
                        })
            
            return web.json_response(presets)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
    
    @PromptServer.instance.routes.get("/vnccs/pose_preset/{filename}")
    async def vnccs_get_pose_preset(request):
        """Get specific pose preset file"""
        try:
            filename = request.match_info.get("filename", "")
            if not filename.endswith('.json'):
                return web.Response(text="Only JSON files allowed", status=400)
            
            presets_dir = os.path.join(os.path.dirname(__file__), "presets", "poses")
            file_path = os.path.join(presets_dir, filename)
            
            # Security check
            if not os.path.abspath(file_path).startswith(os.path.abspath(presets_dir)):
                return web.Response(text="Invalid path", status=400)
            
            if not os.path.exists(file_path):
                return web.Response(text=f"Preset not found: {filename}", status=404)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return web.json_response(data)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    # Character Studio API for real-time 3D preview
    @PromptServer.instance.routes.get("/vnccs/skeleton/get_bones")
    async def vnccs_get_skeleton_bones(request):
        """Get MakeHuman skeleton bone data for Debug2 node"""
        try:
            from .nodes.character_studio import CHARACTER_STUDIO_CACHE, _ensure_data_loaded
            from .CharacterData.mh_parser import HumanSolver
            
            _ensure_data_loaded()
            skel = CHARACTER_STUDIO_CACHE.get('skeleton')
            
            if not skel:
                return web.json_response({"error": "Skeleton not loaded"}, status=500)
            
            # CRITICAL: Update joint positions with current mesh!
            # Otherwise skeleton shows REST pose which doesn't match the morphed mesh
            solver = HumanSolver()
            # Use same factors as the default mesh API call uses
            # age=25.0 -> normalized = (25-1)/(90-1) = 0.27
            # gender=0.5, weight=0.5, muscle=0.5, height=0.5, breast=0.5, genital=0.5
            mh_age = (25.0 - 1.0) / (90.0 - 1.0)  # Same normalization as mesh API
            factors = solver.calculate_factors(mh_age, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
            current_verts = solver.solve_mesh(
                CHARACTER_STUDIO_CACHE['base_mesh'], 
                CHARACTER_STUDIO_CACHE['targets'], 
                factors
            )
            
            # Create mesh wrapper and update skeleton
            class MeshWrapper:
                def __init__(self, verts):
                    self.vertices = verts
            
            mesh_wrapper = MeshWrapper(current_verts)
            skel.updateJointPositions(mesh_wrapper)
            
            bones_data = []
            for bone in skel.getBones():
                headPos = bone.headPos.tolist() if hasattr(bone.headPos, 'tolist') else list(bone.headPos)
                tailPos = bone.tailPos.tolist() if hasattr(bone.tailPos, 'tolist') else list(bone.tailPos)
                
                # Include rest orientation matrix for proper Three.js alignment
                # This matrix defines how the bone is oriented in rest pose
                restMatrix = None
                if bone.matRestGlobal is not None:
                    restMatrix = bone.matRestGlobal.flatten().tolist()
                
                bones_data.append({
                    "name": bone.name,
                    "headPos": headPos,
                    "tailPos": tailPos,
                    "parent": bone.parent.name if bone.parent else None,
                    "length": float(bone.length) if hasattr(bone, 'length') else 0.0,
                    "restMatrix": restMatrix  # 4x4 matrix as flat array
                })
            
            return web.json_response({
                "status": "success",
                "bones": bones_data
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            return web.json_response({"error": str(e)}, status=500)

    @PromptServer.instance.routes.post("/vnccs/character_studio/update_preview")
    async def vnccs_character_studio_update_preview(request):
        try:
            import numpy as np
            import json
            data = await request.json()
            
            # Extract params
            age = float(data.get('age', 25.0))
            gender = float(data.get('gender', 0.5))
            weight = float(data.get('weight', 0.5))
            muscle = float(data.get('muscle', 0.5))
            height = float(data.get('height', 0.5))
            breast_size = float(data.get('breast_size', 0.5))
            genital_size = float(data.get('genital_size', 0.5))
            
            # Optional Pose Data
            pose_data = data.get('pose') # Expects dict {joint: {x,y,z}} called 'joints' or just logic
            manual_pose = data.get('manual_pose') # For debug2: {bone_name: [x,y,z]} (Euler degrees)
            bone_positions = data.get('bone_positions') # For debug2 drag: {bone_name: {x,y,z}}
             
            # Import from CharacterData
            from .CharacterData.mh_parser import TargetParser, HumanSolver
            from .CharacterData.obj_loader import load_obj
            from .CharacterData import matrix
            from .nodes.character_studio import CHARACTER_STUDIO_CACHE, _ensure_data_loaded
            
            # Normalize age
            mh_age = (age - 1.0) / (90.0 - 1.0)
            mh_age = max(0.0, min(1.0, mh_age))
            
            # Ensure data loaded
            _ensure_data_loaded()
            
            # Solve mesh
            solver = HumanSolver()
            factors = solver.calculate_factors(mh_age, gender, weight, muscle, height, breast_size, genital_size)
            new_verts = solver.solve_mesh(CHARACTER_STUDIO_CACHE['base_mesh'], CHARACTER_STUDIO_CACHE['targets'], factors)
            
            # Apply Pose if present
            skel = CHARACTER_STUDIO_CACHE.get('skeleton')
            has_pose = False
            
            if manual_pose and skel:
                # Manual FK Control (Debug2)
                # IMPORTANT: Must update bones in hierarchical order (roots first)
                for bone in skel.getBones():
                    b_name = bone.name
                    if b_name in manual_pose:
                        rot_euler = manual_pose[b_name]
                        if len(rot_euler) == 3:
                            # Create rotation matrix from Euler (Degrees)
                            rx = matrix.rotx(rot_euler[0])
                            ry = matrix.roty(rot_euler[1])
                            rz = matrix.rotz(rot_euler[2])
                            # Order: Z * Y * X
                            mat = np.dot(rz, np.dot(ry, rx))
                            # CRITICAL FIX: Preserve translation to prevent mesh from breaking
                            mat[:3, 3] = np.asarray(bone.matPose[:3, 3]).ravel()
                            bone.matPose = mat
                    # CRITICAL: Call update() to recalculate matPoseGlobal and matPoseVerts
                    bone.update()
                
                # Create mesh wrapper with current vertices for joint position update
                class MeshWrapper:
                    def __init__(self, verts):
                        self.vertices = verts
                
                mesh_wrapper = MeshWrapper(new_verts)
                skel.updateJointPositions(mesh_wrapper)
                has_pose = True
                
            elif pose_data and skel:
                 try:
                     from .nodes.pose_transfer import PoseTransfer
                     if "joints" in pose_data: pose_data = pose_data["joints"]
                     pt = PoseTransfer(skel)
                     pt.apply_pose(pose_data)
                     has_pose = True
                 except Exception as e:
                     print(f"PoseAPI Error: {e}")
            
            # IK Solver Integration
            ik_target = data.get("ik_target")
            if ik_target and skel:
                try:
                    from .nodes.ik_solver import IKSolver
                    bone_name = ik_target.get("bone")
                    target_pos = ik_target.get("pos")
                    chain_len = ik_target.get("chain_length", 3)
                    
                    if bone_name and target_pos:
                        # Solve IK
                        ik_rotations = IKSolver.solve_ik(
                            skel, 
                            bone_name, 
                            target_pos, 
                            chain_length=chain_len,
                            max_iterations=10
                        )
                        
                        # Apply resulting rotations
                        if not manual_pose: manual_pose = {}
                        manual_pose.update(ik_rotations)
                        
                        is_relative = data.get("relative", False)

                        # Apply manual_pose (Standard Logic)
                        for b_name, rot in manual_pose.items():
                            bone = skel.getBone(b_name)
                            if bone:
                                from .CharacterData import transformations as tm
                                import math
                                
                                # Current Rotation if relative
                                if is_relative:
                                    # We need current angles. 
                                    # matPose is local.
                                    cur_x, cur_y, cur_z = tm.euler_from_matrix(bone.matPose, 'rzyx')
                                    rx = math.degrees(cur_x) + rot[0]
                                    ry = math.degrees(cur_y) + rot[1]
                                    rz = math.degrees(cur_z) + rot[2]
                                else:
                                    rx, ry, rz = rot[0], rot[1], rot[2]

                                mat = np.dot(
                                    tm.rotation_matrix(math.radians(rz), [0, 0, 1]),
                                    np.dot(
                                        tm.rotation_matrix(math.radians(ry), [0, 1, 0]),
                                        tm.rotation_matrix(math.radians(rx), [1, 0, 0])
                                    )
                                )
                                # PRESERVE TRANSLATION to avoid tearing/collapse
                                mat[:3, 3] = np.asarray(bone.matPose[:3, 3]).ravel()
                                bone.matPose = mat
                                bone.update()
                                
                        # Final sweep to ensure hierarchical global matrices are correct
                        for b in skel.getBones():
                            b.update()
                                
                        has_pose = True
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"IK Error: {e}")
            
            elif bone_positions and skel:
                # Direct bone position control (Debug2 drag mode)
                # For each moved bone, update its headPos directly
                # This is a simple approach - just move the bone head positions
                class MeshWrapper:
                    def __init__(self, verts):
                        self.vertices = verts
                
                mesh_wrapper = MeshWrapper(new_verts)
                
                for b_name, pos in bone_positions.items():
                    bone = skel.getBone(b_name)
                    if bone:
                        # Calculate displacement from original position
                        orig_head = np.array(bone.headPos)
                        new_head = np.array([pos.get('x', 0), pos.get('y', 0), pos.get('z', 0)])
                        
                        # Create translation matrix
                        displacement = new_head - orig_head
                        bone.matPose = matrix.translate(displacement.tolist())
                
                skel.updateJointPositions(mesh_wrapper)
                has_pose = True
            
            # Shared Skinning Logic
            if has_pose and skel and skel.vertexWeights:
                 try:
                     weights_data = skel.vertexWeights.data
                     deformed_verts = np.zeros_like(new_verts)
                     total_weight = np.zeros((new_verts.shape[0], 1), dtype=np.float32)
                     
                     bone_matrices = {b.name: b.matPoseVerts for b in skel.getBones()}
                     verts_hom = np.hstack([new_verts, np.ones((len(new_verts), 1), dtype=np.float32)])
                     
                     for b_name, (indices, w_vals) in weights_data.items():
                         if b_name not in bone_matrices: continue
                         mat = np.array(bone_matrices[b_name])
                         
                         # Apply transform
                         vs = verts_hom[indices]
                         vs_trans = np.dot(vs, mat.T)
                         w_col = w_vals[:, np.newaxis]
                         
                         deformed_verts[indices] += vs_trans[:, :3] * w_col
                         np.add.at(total_weight, indices, w_col)
                         
                     # Normalize/Fallback
                     # If total_weight > 0, we trust it. If 0, keep rest pose.
                     # Actually LBS usually requires normalized weights. 
                     # MakeHuman weights sum to 1. But our 'extra_mapping' might exceed 1?
                     # No, we just added to orphans.
                     # If sum > 1, mesh explodes/scales.
                     # We should technically normalize. But for Debug, raw LBS is fine for now.
                     
                     unskinned = (total_weight < 0.01).flatten()
                     deformed_verts[unskinned] = new_verts[unskinned]
                     new_verts = deformed_verts
                 except Exception as e:
                     print(f"Skinning Error: {e}")

            # Filter faces and return
            base_mesh = CHARACTER_STUDIO_CACHE['base_mesh']
            valid_prefixes = ["body", "helper-r-eye", "helper-l-eye", "helper-upper-teeth", "helper-lower-teeth", "helper-tongue", "helper-genital"]
            
            valid_faces = []
            if base_mesh.face_groups:
                for i, group in enumerate(base_mesh.face_groups):
                    g_clean = group.strip()
                    is_valid = g_clean in valid_prefixes
                    if g_clean.startswith("joint-"): is_valid = False
                    if g_clean in ["helper-skirt", "helper-tights", "helper-hair"]: is_valid = False
                    if g_clean == "helper-genital" and gender < 0.99: is_valid = False
                    
                    if is_valid:
                        valid_faces.append(base_mesh.faces[i])
            
            # Convert quads to triangles
            tri_indices = []
            for face in valid_faces:
                # Resolve vertex indices (handle both [v, vt, vn] and [v] formats)
                # face[i] can be an int (index) or a tuple/list (index, tex, norm)
                
                v_indices = []
                for item in face:
                    if isinstance(item, (list, tuple)):
                        v_indices.append(item[0])
                    else:
                        v_indices.append(item)
                
                if len(v_indices) == 3:
                     tri_indices.extend([v_indices[0], v_indices[1], v_indices[2]])
                elif len(v_indices) == 4:
                     tri_indices.extend([v_indices[0], v_indices[1], v_indices[2]])
                     tri_indices.extend([v_indices[0], v_indices[2], v_indices[3]])
            
            # Extract Bones Data for Unified Visualizer
            bones_data = []
            if skel:
                # Ensure joints are updated to match the morphed mesh (if not already done by pose logic)
                # We need to do this even if no pose was applied, to match body shape (height, weight etc)
                if not has_pose: # If has_pose is true, it was already updated above
                     class MeshWrapper:
                        def __init__(self, verts):
                            self.vertices = verts
                     mesh_wrapper = MeshWrapper(new_verts)
                     skel.updateJointPositions(mesh_wrapper)

                for bone in skel.getBones():
                    # Get Base Rest Positions
                    headPos = bone.headPos.tolist() if hasattr(bone.headPos, 'tolist') else list(bone.headPos)
                    tailPos = bone.tailPos.tolist() if hasattr(bone.tailPos, 'tolist') else list(bone.tailPos)
                    
                    # Apply Pose Transform for Visualization
                    # We use matPoseVerts which is used for skinning vertices.
                    # It transforms a point from Rest Mesh Space to Posed Mesh Space.
                    # Since headPos/tailPos are in Rest Mesh Space, this is exactly what we need.
                    
                    finalHead = headPos
                    finalTail = tailPos
                    
                    if hasattr(bone, 'matPoseVerts') and bone.matPoseVerts is not None:
                        # matPoseVerts is likely a flat list or numpy array 4x4
                        import numpy as np
                        mat = np.array(bone.matPoseVerts)
                        if mat.shape == (4, 4):
                             # Transform Head
                             h_vec = np.array([headPos[0], headPos[1], headPos[2], 1.0])
                             h_trans = np.dot(mat, h_vec)
                             finalHead = h_trans[:3].tolist()
                             
                             # Transform Tail
                             t_vec = np.array([tailPos[0], tailPos[1], tailPos[2], 1.0])
                             t_trans = np.dot(mat, t_vec)
                             finalTail = t_trans[:3].tolist()
                    
                    restMatrix = None
                    if bone.matRestGlobal is not None:
                        restMatrix = bone.matRestGlobal.flatten().tolist()
                    
                    bones_data.append({
                        "name": bone.name,
                        "headPos": finalHead,
                        "tailPos": finalTail,
                        "parent": bone.parent.name if bone.parent else None,
                        "length": float(bone.length) if hasattr(bone, 'length') else 0.0,
                        "restMatrix": restMatrix
                    })

            # Prepare weights for frontend skinning
            weights_for_frontend = {}
            if skel and skel.vertexWeights:
                for bone_name, (indices, w_vals) in skel.vertexWeights.data.items():
                    weights_for_frontend[bone_name] = {
                        "indices": indices.tolist() if hasattr(indices, 'tolist') else list(indices),
                        "weights": w_vals.tolist() if hasattr(w_vals, 'tolist') else list(w_vals)
                    }

            return web.json_response({
                "status": "success",
                "vertices": new_verts.flatten().tolist(),
                "indices": tri_indices,
                "normals": [],
                "bones": bones_data,
                "weights": weights_for_frontend
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            return web.json_response({"error": str(e)}, status=500)
            for f in valid_faces:
                if len(f) == 4:
                    tri_indices.extend([f[0], f[1], f[2]])
                    tri_indices.extend([f[0], f[2], f[3]])
                elif len(f) == 3:
                    tri_indices.extend([f[0], f[1], f[2]])

            # Calculate normals
            verts_np = new_verts
            normals = np.zeros_like(verts_np)
            
            tris = np.array(tri_indices).reshape(-1, 3)
            v0 = verts_np[tris[:, 0]]
            v1 = verts_np[tris[:, 1]]
            v2 = verts_np[tris[:, 2]]
            
            norms = np.cross(v1 - v0, v2 - v0)
            np.add.at(normals, tris[:, 0], norms)
            np.add.at(normals, tris[:, 1], norms)
            np.add.at(normals, tris[:, 2], norms)
            
            norms_len = np.linalg.norm(normals, axis=1, keepdims=True)
            norms_len[norms_len == 0] = 1.0
            normals = normals / norms_len
            
            return web.json_response({
                "vertices": new_verts.flatten().tolist(),
                "indices": tri_indices,
                "normals": normals.flatten().tolist(),
                "status": "success"
            })
            
        except Exception as e:
            print(f"Character Studio API Error: {e}")
            import traceback as tb
            tb.print_exc()
            return web.json_response({"status": "error", "message": str(e)})

_vnccs_register_endpoint()