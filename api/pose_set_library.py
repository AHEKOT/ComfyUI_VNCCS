"""HTTP adapter for the project-owned pose set library."""

import asyncio
import base64
import json

from aiohttp import web

from ..nodes import pose_sets
from ..utils import validate_privileged_request


def register_routes(routes):
    async def respond(operation):
        try:
            return await asyncio.to_thread(operation)
        except FileNotFoundError as error:
            return web.json_response({"error": str(error)}, status=404)
        except (ValueError, TypeError, KeyError) as error:
            return web.json_response({"error": str(error)}, status=400)
        except Exception as error:
            return web.json_response({"error": str(error)}, status=500)

    @routes.get("/vnccs/pose-sets")
    async def catalog(request):
        return await respond(lambda: web.json_response(pose_sets.catalog()))

    @routes.get("/vnccs/pose-sets/preview-sources")
    async def sources(request):
        return await respond(lambda: web.json_response(pose_sets.preview_sources()))

    @routes.get("/vnccs/pose-sets/export")
    async def export(request):
        return await respond(lambda: web.json_response({"version": 1, "sets": pose_sets.local_sets()},
            headers={"Content-Disposition": 'attachment; filename="pose_sets.json"'}))

    @routes.get("/vnccs/pose-sets/{source}/{id}/preview/{index}")
    async def preview(request):
        def read():
            record = pose_sets.get_set(request.match_info["source"], request.match_info["id"])
            index = int(request.match_info["index"])
            if not 0 <= index < len(record["previews"]):
                raise ValueError("Invalid pose index")
            return web.Response(body=base64.b64decode(record["previews"][index], validate=True), content_type="image/webp")
        return await respond(read)

    @routes.get("/vnccs/pose-sets/{source}/{id}")
    async def get(request):
        return await respond(lambda: web.json_response(pose_sets.get_set(request.match_info["source"], request.match_info["id"])))

    @routes.post("/vnccs/pose-sets/{action}")
    async def mutate(request):
        try:
            validate_privileged_request(request)
        except ValueError as error:
            return web.json_response({"error": str(error)}, status=403)
        try:
            raw = bytearray()
            async for chunk in request.content.iter_chunked(65536):
                raw.extend(chunk)
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("Request must be an object")
        except (ValueError, TypeError) as error:
            return web.json_response({"error": str(error)}, status=400)
        def apply():
            action = request.match_info["action"]
            if action == "save":
                pose_sets.save_set(data)
            elif action == "delete":
                pose_sets.delete_set(data.get("id"), data.get("source", "local"))
            elif action == "repository":
                pose_sets.repository_action(data)
            else:
                raise ValueError("Unknown pose library action")
            try:
                result = pose_sets.catalog()
            except (OSError, ValueError, TypeError, KeyError, RecursionError) as error:
                result = {"errors": [{"path": "PoseSetLibrary",
                    "message": f"Changes saved, but the library could not be refreshed: {error}"}]}
            return web.json_response({"ok": True, **result})
        return await respond(apply)
