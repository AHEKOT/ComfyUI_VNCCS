"""Sprite library endpoints owned exclusively by Pose Manager."""

import asyncio
import json

from aiohttp import web

from ..nodes import pose_manager_sprites as sprites
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

    @routes.get("/vnccs/pose-manager/sprites")
    async def catalog(request):
        return await respond(lambda: web.json_response(sprites.catalog()))

    @routes.get("/vnccs/pose-manager/sprites/{id}/preview")
    async def preview(request):
        def read():
            raw, content_type = sprites.asset(request.match_info["id"], preview=True,
                                               repo_id=request.query.get("repository", sprites.REPOSITORY))
            return web.Response(body=raw, content_type=content_type)
        return await respond(read)

    @routes.get("/vnccs/pose-manager/sprites/{id}")
    async def pose(request):
        return await respond(lambda: web.json_response({"pose": sprites.asset(request.match_info["id"],
            repo_id=request.query.get("repository", sprites.REPOSITORY))}))

    @routes.post("/vnccs/pose-manager/sprites")
    async def update(request):
        try:
            validate_privileged_request(request)
        except ValueError as error:
            return web.json_response({"error": str(error)}, status=403)
        try:
            raw = bytearray()
            async for chunk in request.content.iter_chunked(4096):
                raw.extend(chunk)
                if len(raw) > 4096:
                    return web.json_response({"error": "Request exceeds 4 KiB"}, status=413)
            data = json.loads(raw)
            if not isinstance(data, dict) or data.get("action") not in {"ensure", "add", "refresh", "toggle", "remove"}:
                raise ValueError("Invalid sprite library action")
            request_id = data.get("request_id", "")
            if not isinstance(request_id, str) or len(request_id) > 64:
                raise ValueError("Invalid library request identifier")
        except (ValueError, TypeError) as error:
            return web.json_response({"error": str(error)}, status=400)
        def progress(repo_id, completed, total):
            from server import PromptServer
            PromptServer.instance.send_sync("vnccs.pose_manager.library.progress", {
                "request_id": request_id, "repository": repo_id, "completed": completed, "total": total,
            })
        notify = progress if request_id else None
        return await respond(lambda: web.json_response(sprites.ensure(progress=notify) if data["action"] == "ensure"
            else sprites.repository_action(data["action"], data.get("repo_id", sprites.REPOSITORY), data.get("enabled"), progress=notify)))
