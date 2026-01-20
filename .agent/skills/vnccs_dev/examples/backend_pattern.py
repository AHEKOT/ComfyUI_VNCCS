# Example: Backend Endpoint Pattern
# 
# This example demonstrates how to register a dynamic API endpoint in ComfyUI backend.
# Based on: __init__.py patterns

from server import PromptServer
from aiohttp import web
import json

def register_example_route():
    # 1. Get the existing app routes
    routes = PromptServer.instance.routes

    # 2. Define handler with request binding
    @routes.post("/vnccs/example/action")
    async def example_action(request):
        try:
            # Parse body
            data = await request.json()
            param = data.get("param", "default")
            
            # Perform server-side logic
            result = {"status": "success", "echo": param}
            
            # Return JSON response
            return web.json_response(result)
            
        except Exception as e:
            # Handle errors gracefully
            return web.json_response({"status": "error", "message": str(e)}, status=500)

# 3. Call registration function at module level
register_example_route()
