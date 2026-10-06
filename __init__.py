"""VNCCS - Visual Novel Character Creator Suite for ComfyUI."""

import importlib.util
import os, json
import sys
import traceback

def _runtime_module_available(name):
    if name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False

if (
    _runtime_module_available("torch")
    and _runtime_module_available("comfy")
    and _runtime_module_available("folder_paths")
):
    _nodes_module_name = f"{__name__}.nodes"
    _preloaded_nodes = sys.modules.get(_nodes_module_name)
    if _preloaded_nodes is not None and getattr(_preloaded_nodes, "__file__", None) is None:
        # Some custom-node loaders may pre-register this path as a namespace package.
        # Remove that placeholder so the regular relative import executes nodes/__init__.py.
        sys.modules.pop(_nodes_module_name, None)

    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    if not NODE_CLASS_MAPPINGS:
        raise RuntimeError("VNCCS node registration returned no nodes. Check the preceding registration traceback.")
else:
    # Package discovery and source-analysis tools do not provide a ComfyUI runtime.
    # Keep importing metadata possible without importing model dependencies such as torch.
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

WEB_DIRECTORY = "web"

def _vnccs_register_endpoint():  # lazy registration to avoid import errors in analysis tools
    try:
        from server import PromptServer
        from aiohttp import web
    except Exception:
        return

    if getattr(PromptServer.instance, "app", None) is not None:
        from .nodes.http_state import install_cache_policy
        install_cache_policy(PromptServer.instance)

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

    @PromptServer.instance.routes.post("/vnccs/create")
    @PromptServer.instance.routes.get("/vnccs/create")
    async def vnccs_create_character(request):
        try:
            from .utils import validate_privileged_request
            validate_privileged_request(request)
        except ValueError as error:
            return web.json_response({"error": str(error)}, status=403)
        if getattr(request, "method", "GET") == "POST":
            try:
                data = await request.json()
                name = str(data.get("name", "")).strip()
            except (ValueError, TypeError, AttributeError):
                return web.json_response({"error": "Invalid create request"}, status=400)
        else:
            # Compatibility for older extensions; responses are explicitly no-store.
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
        )
        if getattr(request, "method", "GET") == "POST" and data.get("catalog") == "creator_v2":
            defaults["hair"] = "black hair, waist-length hair"
        try:
            from .utils import base_output_dir, load_config, create_initial_character
            base_path = base_output_dir()
            os.makedirs(base_path, exist_ok=True)
            existing_data = load_config(name, strict=True)
            if existing_data is not None:
                return web.json_response({
                    "ok": True,
                    "name": name,
                    "existing": True,
                    "data": existing_data,
                })
            positive_prompt, seed, negative_prompt, age_lora_strength, _sheets_path, _faces_path, face_details = create_initial_character(**defaults)
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

_vnccs_register_endpoint()
