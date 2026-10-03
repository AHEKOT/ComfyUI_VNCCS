"""Small HTTP helpers shared by VNCCS widgets; independent of model runtimes."""


async def prevent_runtime_cache(request, response):
    """Do not let browser/proxy caches reuse mutable extension responses."""
    if request.path.startswith(("/vnccs/", "/api/vnccs/")):
        response.headers["Cache-Control"] = "no-store, private"
        response.headers["Pragma"] = "no-cache"


def install_cache_policy(server):
    app = getattr(server, "app", None)
    if app is not None and not getattr(server, "_vnccs_cache_policy", False):
        app.on_response_prepare.append(prevent_runtime_cache)
        server._vnccs_cache_policy = True
