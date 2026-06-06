import folder_paths
import os
import json
import torch
from server import PromptServer

class VNCCS_AdvancedPlyViewer:
    """
    Advanced PLY Viewer with Custom Camera Control.
    
    Provides an interactive 3D WebGL viewer with mouse flight controls,
    precise coordinate sliders, and the ability to save/load camera states.
    Exports the exact rendered view as an IMAGE tensor.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ply_data": ("PLY_DATA",),
                "output_width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8, "tooltip": "Final render width"}),
                "output_height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8, "tooltip": "Final render height"}),
                # Hidden inputs managed by JS
                "camera_state": ("STRING", {"default": "{}", "multiline": True}),
                "saved_cameras": ("STRING", {"default": "[]", "multiline": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "EXTRINSICS", "INTRINSICS")
    RETURN_NAMES = ("IMAGE", "camera_poses", "camera_intrinsics")
    OUTPUT_NODE = True
    FUNCTION = "render_view"
    CATEGORY = "VNCCS/3D"
    
    def render_view(self, ply_data, output_width, output_height, camera_state="{}", saved_cameras="[]"):
        from ..background_data.worldmirror.src.utils.fast_ply_render import FastPLYRenderer
        
        # 1. Parse Camera State
        try:
            state = json.loads(camera_state) if camera_state else {}
        except Exception:
            state = {}
            
        print(f"🎬 [VNCCS_AdvancedPlyViewer] Received Camera State: {state}")
        
        # 2. Extract specific view if provided in state (extrinsics/intrinsics from frontend)
        # If no state is provided from UI (e.g. first run or API call without interaction),
        # use the first camera from ply_data.
        
        extrinsics_tgt = None
        intrinsics_tgt = None
        
        if "extrinsics" in state and "intrinsics" in state:
            # Reconstruct tensors from JS arrays
            try:
                ex_arr = state["extrinsics"] # Should be 4x4 array
                in_arr = state["intrinsics"] # Should be 3x3 array
                
                # Expand dims to match standard [B, 1, 4, 4] format used in pipeline
                extrinsics_tgt = torch.tensor(ex_arr, dtype=torch.float32).view(1, 1, 4, 4)
                intrinsics_tgt = torch.tensor(in_arr, dtype=torch.float32).view(1, 1, 3, 3)
                print("   ✅ Using custom camera coordinates from UI")
            except Exception as e:
                print(f"   ⚠️ Error parsing custom camera state: {e}")
        
        if extrinsics_tgt is None or intrinsics_tgt is None:
            # Fallback to first available camera in ply_data
            if "camera_poses" in ply_data and ply_data["camera_poses"].shape[1] > 0:
                extrinsics_tgt = ply_data["camera_poses"][:, 0:1, :, :]
                intrinsics_tgt = ply_data["camera_intrinsics"][:, 0:1, :, :]
                print("   ℹ️ Falling back to First View (index 0) from PLY data")
            else:
                raise ValueError("No camera state from UI and no camera parameters found in ply_data.")
                
        # 3. Render the specific view
        print(f"   📷 Rendering {output_width}x{output_height} view...")
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # We need to construct the rendering args
        try:
            renderer = FastPLYRenderer(
                sh_degree=3 if 'sh_features' in ply_data else 0,
                device=device
            )
            
            # Use background_color from state if available, otherwise match original
            bg_color = state.get("background_color", [1.0, 1.0, 1.0])
            bg_tensor = torch.tensor(bg_color, dtype=torch.float32, device=device)
            
            render_result = renderer.render_from_dict(
                splat_data=ply_data,
                extrinsics=extrinsics_tgt.to(device),
                intrinsics=intrinsics_tgt.to(device),
                image_width=output_width,
                image_height=output_height,
                background_color=bg_tensor
            )
            
            image_tensor = render_result["image"] # [B, S, C, H, W]
            
            # ComfyUI expects [B, H, W, C]
            # S is usually 1 here
            image_out = image_tensor[0, 0].permute(1, 2, 0).cpu().unsqueeze(0) # [1, H, W, 3]
            
        except ImportError:
            print("[VNCCS_AdvancedPlyViewer] FastPLYRenderer not available. Cannot render. Returning empty image.")
            image_out = torch.zeros((1, output_height, output_width, 3))
        
        # Important: To show the UI preview AND return the rendered image, 
        # we need to pass the PLY data path to the frontend.
        # Check if python ply_data contains the original file path.
        ply_path = ""
        filename = ""
        if "metadata" in ply_data and "source_file" in ply_data["metadata"]:
            ply_path = ply_data["metadata"]["source_file"]
            filename = os.path.basename(ply_path)
            
        # Send UI update (this triggers the JS viewer to reload if the file changed)
        # We need to handle relative paths for viewing
        rel_path = ""
        output_dir = folder_paths.get_output_directory().replace("\\", "/")
        if ply_path and ply_path.replace("\\", "/").startswith(output_dir):
            rel_path = os.path.relpath(ply_path, output_dir).replace("\\", "/")
            
        ui_data = {
            "filename": [filename],
            "ply_path": [rel_path],
            "type": ["output"],
            "state_ack": [state.get("update_id", "")] # Echo back to JS to confirm we read it
        }
        
        return {"ui": ui_data, "result": (image_out, extrinsics_tgt, intrinsics_tgt)}


NODE_CLASS_MAPPINGS = {
    "VNCCS_AdvancedPlyViewer": VNCCS_AdvancedPlyViewer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCS_AdvancedPlyViewer": "👁️ Advanced PLY Viewer",
}
        
