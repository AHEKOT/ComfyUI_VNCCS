"""
Server extension for serving VNCCS style static files
"""

import os
import json
from typing import Dict, Any

try:
    import server
    from ..utils import load_config, save_config
    
    # Styles folder used by the old VNCCS stylepicker has been removed.
    # No static files are registered by this extension anymore.
    print("[VNCCS] Styles support disabled (stylepicker removed)")
    
except ImportError:
    print("[VNCCS] ComfyUI server not found, static files not registered")
except Exception as e:
    print(f"[VNCCS] Error registering static files: {e}")
