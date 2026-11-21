"""Read-only HTTP catalog for the Additional Poses selector."""

import asyncio

from aiohttp import web

from ..nodes.additional_poses import catalog
from ..nodes.character_generator import _view_url_for_output_path


def register_routes(routes):
    @routes.get("/vnccs/additional_poses/catalog")
    async def get_catalog(request):
        def read():
            result = catalog(request.query.get("character", ""), request.query.get("costume", ""))
            if "emotions" in result:
                result["emotions"] = [
                    {"name": name, "sprites": [
                        {"name": path.name, "url": _view_url_for_output_path(path)} for path in paths
                    ]} for name, paths in result["emotions"].items()
                ]
            return web.json_response(result)

        try:
            return await asyncio.to_thread(read)
        except FileNotFoundError as error:
            return web.json_response({"error": str(error)}, status=404)
        except ValueError as error:
            return web.json_response({"error": str(error)}, status=400)
        except OSError as error:
            return web.json_response({"error": str(error)}, status=500)
