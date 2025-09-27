"""
Server extension for serving VNCCS style static files
"""

import os
import json
from typing import Dict, Any

try:
    import server
    from ..utils import load_config, save_config
    
    styles_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "styles")
    
    if os.path.exists(styles_path):
        server.PromptServer.instance.app.router.add_static(
            '/extensions/VNCCS/styles/', 
            styles_path
        )
        print(f"[VNCCS] Static style files registered: {styles_path}")
        
        png_files = [f for f in os.listdir(styles_path) if f.endswith('.png')]
        print(f"[VNCCS] Found {len(png_files)} PNG style files")
    else:
        print(f"[VNCCS] ERROR: Styles folder not found: {styles_path}")
    
except ImportError:
    print("[VNCCS] ComfyUI server not found, static files not registered")
except Exception as e:
    print(f"[VNCCS] Error registering static files: {e}")
