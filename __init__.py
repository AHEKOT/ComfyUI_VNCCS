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
    @PromptServer.instance.routes.post("/vnccs/character_studio/update_preview")
    async def vnccs_character_studio_update_preview(request):
        try:
            import numpy as np
            data = await request.json()
            
            # Extract params
            age = float(data.get('age', 25.0))
            gender = float(data.get('gender', 0.5))
            weight = float(data.get('weight', 0.5))
            muscle = float(data.get('muscle', 0.5))
            height = float(data.get('height', 0.5))
            breast_size = float(data.get('breast_size', 0.5))
            genital_size = float(data.get('genital_size', 0.5))
            
            # Import from CharacterData
            from .CharacterData.mh_parser import TargetParser, HumanSolver
            from .CharacterData.obj_loader import load_obj
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