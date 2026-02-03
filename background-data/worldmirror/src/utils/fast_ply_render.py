
import torch
import torch.nn.functional as F
import numpy as np
import time
import math

class FastPLYRenderer:
    """
    Super-fast PyTorch Point Cloud Renderer.
    Renders millions of points by projecting them and using scatter operations.
    No loops, no OpenGL, no custom CUDA.
    """
    
    def __init__(self, device):
        self.device = device

    def render(self, means, colors, opacities, scales, c2w, width, height, fov_deg=90.0, bg_color=(0.0, 0.0, 0.0)):
        """
        Render generic point cloud / splats via screen-space scattering.
        
        Args:
            means: [N, 3]
            colors: [N, 3] (0-1)
            opacities: [N] (0-1)
            scales: [N, 3] - used to determine point size
            c2w: [4, 4] camera-to-world
            width, height: resolution
        """
        # 1. Setup Camera
        fov_rad = math.radians(fov_deg)
        focal = width / (2 * math.tan(fov_rad / 2))
        cx, cy = width / 2, height / 2
        
        w2c = torch.linalg.inv(c2w).to(self.device).float()
        
        # 2. Project Points (World -> Camera)
        # Add homogeneous coord
        ones = torch.ones(means.shape[0], 1, device=self.device)
        points_h = torch.cat([means, ones], dim=1)
        
        # Transform
        points_cam = points_h @ w2c.T
        x, y, z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
        
        # 3. Filter points behind camera
        valid_mask = z > 0.1
        if not valid_mask.any():
            return torch.zeros((height, width, 3), device=self.device)
            
        x = x[valid_mask]
        y = y[valid_mask]
        z = z[valid_mask]
        colors = colors[valid_mask]
        opacities = opacities[valid_mask]
        
        # Use max scale as approximate radius proxy
        if scales is not None:
             scales = scales[valid_mask].max(dim=1).values
        else:
             scales = torch.ones_like(z) * 0.01

        # 4. Project to Screen (Camera -> Image)
        # OpenCV convention: x right, y down. But Pytorch/OpenGL might differ.
        # We assume standard pinhole:
        # u = fx * x / z + cx
        # v = fy * y / z + cy
        # Note: If y is UP in camera space (OpenGL), we need to flip.
        # Our `_make_look_at_pose` uses OpenCV convention (Y down), so no flip needed?
        # Let's verify: In OpenCV, Y is down. +Y -> Bottom.
        
        u = (focal * x / z) + cx
        v = (focal * y / z) + cy
        
        # 5. Screen Bounds Check
        h_mask = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        
        u = u[h_mask]
        v = v[h_mask]
        z = z[h_mask]
        colors = colors[h_mask]
        opacities = opacities[h_mask]
        scales = scales[h_mask]
        
        if u.numel() == 0:
            return torch.zeros((height, width, 3), device=self.device)

        # 6. Sort by Depth (Back-to-Front)
        # We want farthest points drawn first.
        # z is positive (distance). Large z = far.
        # argsort(z, descending=True) -> Far to Near.
        indices = torch.argsort(z, descending=True)
        
        u = u[indices]
        v = v[indices]
        colors = colors[indices]
        opacities = opacities[indices]
        scales = scales[indices]
        z = z[indices] # Need sorted z for point size calc
        
        # 7. Compute Point Size
        # Projected size in pixels ~= scale * focal / z
        sizes = (scales * focal / z).clamp(min=3.0)
        
        # 8. Draw!
        # Since we cannot loop, we use "Scatter Splatting".
        # We round coordinates to integers.
        # To handle occlusion properly with Painter's Algorithm in a vectorized way:
        # We can just assign values to a flat buffer. Latest writes win.
        # Since we sorted Far-to-Near, the Near points will be written last. Perfect.
        
        ui = u.long()
        vi = v.long()
        
        # Create Canvas
        canvas = torch.zeros((height * width, 3), device=self.device)
        depth_canvas = torch.zeros((height * width), device=self.device)
        mask_canvas = torch.zeros((height * width), device=self.device)
        
        # Linear indices
        flat_idx = vi * width + ui
        
        # Simple Points (1px)
        # canvas.index_copy_(0, flat_idx, colors)
        # But this doesn't blend.
        
        # "Splatting" with variable kernel size is hard vectorized.
        # Workaround: Render 3 copies with offsets (Center, Right, Down, Diag) to simulate 2x2 block?
        # Better: Filter by point size. 
        # Large points (> 2px) -> Draw 9 pixels (3x3).
        # Small points -> Draw 1 pixel.
        
        # Let's try Multi-Pass Scatter for "Fat Points" (approx 3x3 kernel)
        
        offsets = [
            (0,0), (1,0), (-1,0), (0,1), (0,-1),
             (1,1), (-1,-1), (1,-1), (-1,1)
        ]
        
        # Start with valid pixels
        base_mask = (ui >= 1) & (ui < width-1) & (vi >= 1) & (vi < height-1) # Safe margin
        
        final_colors = colors * opacities.unsqueeze(1) # Pre-multiply? No, just use as solid color for now
        
        # We just write to canvas.
        # We can implement simple alpha blending? 
        # Standard: Dest = Src * Alpha + Dest * (1 - Alpha).
        # Vectorized sequential modification is tricky in Pytorch (index_put is non-deterministic or atomic sum).
        # BUT! We just want to "paint". Overwriting is fine for "Solid Splats".
        # For opacity, we can just assume solid (alpha=1) for simplicity and speed.
        # The user wants "Backgrounds", usually solid geometry.
        
        # Optimization: Only draw 'fat' splats for points that have projected size > 1.5
        is_large = sizes > 1.5
        
        # Global canvas update helper
        def splat_at(du, dv, point_mask=None):
            if point_mask is not None:
                mask = base_mask & point_mask
            else:
                mask = base_mask
                
            if not mask.any(): return
            
            idx = (vi[mask] + dv) * width + (ui[mask] + du)
            # Use simple assignment for "last writer wins" (Nearest-Front due to sorting)
            canvas[idx] = colors[mask]

        # Draw Center (all points)
        splat_at(0, 0)
        
        # Draw neighbors for large points (3x3 block)
        # This makes points look like 3x3 squares. Faster than circles.
        
        neighbor_offsets = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]
        for dx, dy in neighbor_offsets:
             splat_at(dx, dy, is_large)
             
        # Reshape
        img = canvas.reshape(height, width, 3)
        
        # Simple hole filling (Avg pool)
        # If output has too many black holes, we can dilate.
        # Let's return raw for now.
        
        # Reshape
        img = canvas.reshape(height, width, 3)
        
        # Flip vertically (fix for OpenCV/OpenGL coordinate mismatch)
        img = torch.flip(img, [0])
        
        return img 

