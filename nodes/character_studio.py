"""VNCCS Character Studio - Visual character designer based on MakeHuman."""

import os
import torch
import numpy as np
from PIL import Image, ImageDraw

# Import from CharacterData module
from ..CharacterData.mh_parser import TargetParser, HumanSolver
from ..CharacterData.obj_loader import Mesh, load_obj
from ..CharacterData.mesh_processing import subdivide_catmull_clark_approx
from ..CharacterData.mh_skeleton import Skeleton

# Singleton storage for loaded MH data to avoid reloading every time
CHARACTER_STUDIO_CACHE = {
    "base_mesh": None,
    "targets": None,
    "parser": None,
    "skeleton": None
}


def _get_character_data_path():
    """Get the path to CharacterData folder."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "CharacterData"))


def _ensure_data_loaded():
    """Load MakeHuman data if not already loaded."""
    if CHARACTER_STUDIO_CACHE['base_mesh'] is not None:
        return

    char_data_path = _get_character_data_path()
    mh_path = os.path.join(char_data_path, "makehuman")
    
    if not os.path.exists(mh_path):
        raise Exception(f"MakeHuman data not found at: {mh_path}")

    print(f"[VNCCS Character Studio] Loading MakeHuman data from {mh_path}...")

    # 1. Load Base Mesh
    base_obj_paths = [
        os.path.join(mh_path, "makehuman", "data", "3dobjs", "base.obj"),
        os.path.join(mh_path, "data", "3dobjs", "base.obj"),
    ]
    
    base_path = next((p for p in base_obj_paths if os.path.exists(p)), None)
    if not base_path:
        raise Exception("Could not find base.obj inside makehuman data.")

    CHARACTER_STUDIO_CACHE['base_mesh'] = load_obj(base_path)
    
    # 2. Load Targets
    parser = TargetParser(mh_path)
    CHARACTER_STUDIO_CACHE['targets'] = parser.scan_targets()
    CHARACTER_STUDIO_CACHE['parser'] = parser
    
    print(f"[VNCCS Character Studio] Loaded {len(CHARACTER_STUDIO_CACHE['targets'])} targets.")
    
    # 3. Load Skeleton (Preference: game_engine > default)
    # Check for game_engine.mhskel first (User provided)
    skel_path = os.path.join(mh_path, "makehuman", "data", "rigs", "game_engine.mhskel")
    if not os.path.exists(skel_path):
        skel_path = os.path.join(mh_path, "makehuman", "data", "rigs", "default.mhskel")
        
    if os.path.exists(skel_path):
        print(f"[VNCCS Character Studio] Loading skeleton from {skel_path}...")
        skel = Skeleton()
        skel.fromFile(skel_path, CHARACTER_STUDIO_CACHE['base_mesh'])
        CHARACTER_STUDIO_CACHE['skeleton'] = skel
    else:
        print(f"[VNCCS Character Studio] Warning: Default skeleton not found at {skel_path}")


class VNCCS_CharacterStudio:
    """
    Visual character designer node with real-time 3D preview.
    Based on MakeHuman parametric human model.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "age": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 90.0, "step": 1.0, "label": "Age (Years)"}),
                "gender": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "label": "Gender (0=Fem, 1=Male)"}),
                "weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "muscle": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "height": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "breast_size": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "genital_size": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "VNCCS/Character"

    def generate(self, age, gender, weight, muscle, height, breast_size, genital_size):
        # 1. Ensure data is loaded
        _ensure_data_loaded()
        
        # 2. Normalize age for MakeHuman (0.0 - 1.0)
        mh_age = (age - 1.0) / (90.0 - 1.0)
        mh_age = max(0.0, min(1.0, mh_age))
        
        # 3. Calculate factors and solve mesh
        solver = HumanSolver()
        factors = solver.calculate_factors(mh_age, gender, weight, muscle, height, breast_size, genital_size)
        
        new_verts = solver.solve_mesh(
            CHARACTER_STUDIO_CACHE['base_mesh'],
            CHARACTER_STUDIO_CACHE['targets'],
            factors
        )
        
        # 4. Create mesh copy with new vertices
        mesh = CHARACTER_STUDIO_CACHE['base_mesh'].copy()
        mesh.vertices = new_verts
        
        # 5. Filter faces (remove helpers/joints)
        filtered_faces = []
        if mesh.face_groups:
            valid_prefixes = [
                "body", "helper-r-eye", "helper-l-eye", 
                "helper-upper-teeth", "helper-lower-teeth", 
                "helper-tongue", "helper-genital"
            ]
            
            for i, group in enumerate(mesh.face_groups):
                g_clean = group.strip()
                is_valid = g_clean in valid_prefixes
                
                # Exclude specific groups
                if g_clean.startswith("joint-"):
                    is_valid = False
                if g_clean in ["helper-skirt", "helper-tights", "helper-hair"]:
                    is_valid = False
                    
                # Conditional genitals (only show for male)
                if g_clean == "helper-genital" and gender < 0.99:
                    is_valid = False

                if is_valid:
                    filtered_faces.append(mesh.faces[i])
            
            if len(filtered_faces) > 0:
                mesh.faces = filtered_faces
        
        # 6. Render to image (1536x1536)
        img = self._render_mesh(mesh)
        
        # 7. Convert to tensor
        img_np = np.array(img).astype(np.float32) / 255.0
        return (torch.from_numpy(img_np)[None,],)
    
    def _render_mesh(self, mesh, width=1536, height=1536):
        """Render mesh to PIL Image with flat shading."""
        img = Image.new("RGB", (width, height), (40, 40, 40))
        draw = ImageDraw.Draw(img)
        
        verts = mesh.vertices.copy()
        faces = mesh.faces
        
        # Auto-center
        min_v = verts.min(axis=0)
        max_v = verts.max(axis=0)
        center = (min_v + max_v) / 2
        size = max_v - min_v
        max_dim = np.max(size)
        
        verts -= center
        
        # Scale to fit 90%
        if max_dim < 0.1:
            max_dim = 1.0
        scale = (height * 0.9) / max_dim
        
        verts_screen = verts * scale
        verts_screen[:, 0] += width / 2
        verts_screen[:, 1] = height / 2 - verts_screen[:, 1]
        
        # Render with flat shading
        self._render_flat_shaded(draw, verts_screen, verts, faces, width, height)
        
        return img
    
    def _render_flat_shaded(self, draw, verts_screen, verts_3d, faces, W, H):
        """Pure Python flat shading rasterizer."""
        if isinstance(faces, list):
            try:
                faces_arr = np.array(faces, dtype=np.int32)
            except:
                print("Warning: Mixed face types detected. Preview might be glitchy.")
                return
        else:
            faces_arr = faces

        # Get vertices for each face
        v3d = verts_3d[faces_arr]
        
        # Calculate face normals
        p0 = v3d[:, 0, :]
        p1 = v3d[:, 1, :]
        p2 = v3d[:, 2, :]
        
        edge1 = p1 - p0
        edge2 = p2 - p0
        
        normals = np.cross(edge1, edge2)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms < 1e-6] = 1.0
        normals /= norms
        
        # Lighting (3-point studio setup)
        vis_dot = normals[:, 2]
        
        # Key light (top-left, warm)
        light_pos = np.array([-0.5, 0.8, 1.0])
        light_pos /= np.linalg.norm(light_pos)
        
        # Fill light (right, neutral)
        light_fill = np.array([0.5, 0.2, 0.5])
        light_fill /= np.linalg.norm(light_fill)
        
        diffuse_key = np.maximum(0.0, np.sum(normals * light_pos, axis=1))
        diffuse_fill = np.maximum(0.0, np.sum(normals * light_fill, axis=1))
        
        # Combined intensity
        intensity = 0.3 + 0.6 * diffuse_key + 0.3 * diffuse_fill
        
        # Backface culling
        valid_mask = vis_dot > -0.1
        
        faces_vis = faces_arr[valid_mask]
        v3d_vis = v3d[valid_mask]
        intensity_vis = intensity[valid_mask]
        
        # Z-sorting
        z_avg = v3d_vis[:, :, 2].mean(axis=1)
        sort_idx = np.argsort(z_avg)
        
        # Get screen coordinates
        v_scr = verts_screen[faces_vis]
        
        # Skin tone base color
        base_color = np.array([210, 160, 140])
        
        for i in sort_idx:
            diffuse = intensity_vis[i]
            
            r = min(255, int(210 * diffuse))
            g = min(255, int(160 * diffuse))
            b = min(255, int(140 * diffuse))
            
            f_v = v_scr[i]
            pts = [
                (f_v[0, 0], f_v[0, 1]),
                (f_v[1, 0], f_v[1, 1]),
                (f_v[2, 0], f_v[2, 1]),
                (f_v[3, 0], f_v[3, 1])
            ]
            
            draw.polygon(pts, fill=(r, g, b))


# Node mappings
NODE_CLASS_MAPPINGS = {
    "VNCCS_CharacterStudio": VNCCS_CharacterStudio
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCS_CharacterStudio": "VNCCS Character Studio"
}
