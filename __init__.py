"""VNCCS - Visual Novel Character Creator Suite for ComfyUI."""

import os, json, inspect
import traceback

def _vnccs_auto_migrate():
    """Silently migrate legacy VN_CharacterCreatorSuit → VNCCS/Characters on startup."""
    try:
        from .utils import migrate_legacy_data
        result = migrate_legacy_data()
        if result.get("migrated"):
            names = ", ".join(result.get("details", []))
            print(f"[VNCCS] Auto-migrated {result['count']} character(s) to VNCCS/Characters: {names}")
        elif result.get("error"):
            print(f"[VNCCS] Migration error: {result['error']}")
    except Exception as e:
        print(f"[VNCCS] Migration skipped: {e}")

print("[VNCCS] Automatic legacy migration is disabled. Use the VNCCS Migration Assistent node to migrate legacy sheets.")

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']



WEB_DIRECTORY = "web"

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
            from .utils import config_path, ensure_safe_name
            name = ensure_safe_name(name, "character")
            cfg_path = config_path(name)
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=400)

        if not os.path.exists(cfg_path):
            return web.json_response({"error": "not found"}, status=404)

        try:
            with open(cfg_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return web.json_response(data)
        except Exception as e:
            return web.json_response({"error": "read failed", "detail": str(e)}, status=500)



    @PromptServer.instance.routes.post("/vnccs/migrate")
    async def vnccs_migrate(request):
        """Trigger migration of legacy data to new folder structure."""
        try:
            from .utils import migrate_legacy_data
            result = migrate_legacy_data()
            return web.json_response(result)
        except Exception as e:
            traceback.print_exc()
            return web.json_response({
                "migrated": False,
                "error": str(e),
            }, status=500)

    @PromptServer.instance.routes.post("/vnccs/delete")
    async def vnccs_delete_character(request):
        try:
            from .utils import validate_privileged_request
            validate_privileged_request(request)
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=403)

        try:
            data = await request.json()
        except Exception:
            data = {}
        name = str(data.get("name") or request.rel_url.query.get("name", "")).strip()
        if not name:
            return web.json_response({"error": "name required"}, status=400)
        
        try:
            from .utils import character_dir, safe_join_under, base_output_dir, ensure_safe_name
            import shutil
            name = ensure_safe_name(name, "character")
            char_path = character_dir(name)
            safe_join_under(base_output_dir(), os.path.relpath(char_path, base_output_dir()))
            if not os.path.exists(char_path):
                return web.json_response({"error": f"Character '{name}' not found"}, status=404)
                
            # Permanent Delete as requested
            shutil.rmtree(char_path)
            
            return web.json_response({"ok": True, "name": name, "deleted": True})
            
        except Exception as e:
            traceback.print_exc()
            return web.json_response({
                "error": "delete failed",
                "detail": str(e),
            }, status=500)

    @PromptServer.instance.routes.get("/vnccs/create")
    async def vnccs_create_character(request):
        name = request.rel_url.query.get("name", "").strip()
        if not name:
            return web.json_response({"error": "name required"}, status=400)
        try:
            from .utils import ensure_safe_name
            name = ensure_safe_name(name, "character")
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=400)
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
            from .utils import base_output_dir, safe_join_under
            cc = CharacterCreator()
            base_path = base_output_dir()
            os.makedirs(base_path, exist_ok=True)
            base_char_dir = safe_join_under(base_path, name)
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
                    "data": existing_data,
                })
            # Backward compatibility: drop force_new if method doesn't accept it
            try:
                sig = inspect.signature(cc.create_character)
                if 'force_new' not in sig.parameters and 'force_new' in defaults:
                    defaults.pop('force_new')
            except Exception:
                defaults.pop('force_new', None)
            positive_prompt, seed, negative_prompt, age_lora_strength, _sheets_path, _faces_path, face_details = cc.create_character(**defaults)
            return web.json_response({
                "ok": True,
                "name": name,
                "seed": seed,
                "positive_prompt": positive_prompt,
                "negative_prompt": negative_prompt,
                "age_lora_strength": age_lora_strength,
                "face_details": face_details,
            })
        except Exception as e:
            traceback.print_exc()
            return web.json_response({
                "error": "create failed",
                "detail": str(e),
                "type": type(e).__name__,
            }, status=500)

    @PromptServer.instance.routes.get("/vnccs/create_costume")
    async def vnccs_create_costume(request):
        character_name = request.rel_url.query.get("character", "").strip()
        costume_name = request.rel_url.query.get("costume", "").strip()
        if not character_name or not costume_name:
            return web.json_response({"error": "character and costume required"}, status=400)
        try:
            from .utils import ensure_safe_name
            character_name = ensure_safe_name(character_name, "character")
            costume_name = ensure_safe_name(costume_name, "costume")
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=400)
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
        if os.path.commonpath([os.path.abspath(models_dir), os.path.abspath(file_path)]) != os.path.abspath(models_dir):
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
            if os.path.commonpath([os.path.abspath(presets_dir), os.path.abspath(file_path)]) != os.path.abspath(presets_dir):
                return web.Response(text="Invalid path", status=400)
            
            if not os.path.exists(file_path):
                return web.Response(text=f"Preset not found: {filename}", status=404)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return web.json_response(data)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)



_vnccs_register_endpoint()
