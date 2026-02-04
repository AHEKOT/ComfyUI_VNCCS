
import unittest
import sys
import os
import torch
import numpy as np
from unittest.mock import MagicMock, patch

# Mock ComfyUI folder_paths if not available
try:
    import folder_paths
except ImportError:
    # Create a dummy folder_paths module
    folder_paths = MagicMock()
    folder_paths.models_dir = "/tmp/comfyui/models"
    sys.modules["folder_paths"] = folder_paths

# Mock heavy dependencies globally
# We also mock internal worldplay implementation modules that are heavy or complex
# to isolate the node logic testing.
mock_modules = [
    "diffusers", 
    "transformers", 
    "huggingface_hub",
    "worldplay.inference.pipeline_worldplay",
    "worldplay.models.dits.arwan_w_action"
]

for mod in mock_modules:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "background-data"))

# Now import the modules to test
# We need to mock some heavy imports inside the modules if we want to avoid loading them,
# but usually python imports the top level. 
# wan_transformer import might fail if dependencies are missing, but let's try importing
# the utils first which should be lighter.

from worldplay.utils.trajectory import parse_pose_string, pose_to_input

class TestTrajectoryUtils(unittest.TestCase):
    def test_parse_pose_string_simple(self):
        pose = "w-10"
        motions = parse_pose_string(pose)
        self.assertEqual(len(motions), 10)
        self.assertEqual(motions[0]["forward"], 0.08)

    def test_parse_pose_string_complex(self):
        pose = "w-5, right-5"
        motions = parse_pose_string(pose)
        self.assertEqual(len(motions), 10)
        # first 5 should be forward
        for i in range(5):
            self.assertIn("forward", motions[i])
        # next 5 should be yaw (right)
        for i in range(5, 10):
            self.assertIn("yaw", motions[i])
            self.assertTrue(motions[i]["yaw"] > 0)

    def test_pose_to_input(self):
        # pose_to_input(pose_data, latent_num)
        # 1 latent corresponds to how many frames? In the code it says:
        # latent_num_from_pose = len(pose_keys)
        # assert latent_num_from_pose == latent_num
        
        # So if we have "w-4" (4 frames of movement), we expect 4 latents? 
        # trajectory.py logic:
        # motions = parse_pose_string
        # poses = generate_camera_trajectory_local(motions)
        # pose_json has len(poses) entries.
        # generate_camera_trajectory_local appends T.copy() AFTER each move.
        # And initializes with identity.
        # So for N moves, we have N+1 poses.
        
        pose = "w-4" # 4 frames duration
        # expected latents: 4 motions -> 5 poses (0 to 4) ??
        # Let's check generate_camera_trajectory_local behavior
        # poses = [Identity]
        # loop motions: poses.append(new_T)
        # So "w-4" -> 4 motions -> 5 poses.
        
        # pose_to_input expects latent_num to match json keys count.
        # So we should pass 5.
        
        w2c, K, action = pose_to_input(pose, 5)
        
        self.assertEqual(w2c.shape, (5, 4, 4))
        self.assertEqual(K.shape, (5, 3, 3))
        self.assertEqual(action.shape, (5,))

class TestNodesStructure(unittest.TestCase):
    def setUp(self):
        # We need to patch imports that might trigger actual model loading or heavy imports 
        # inside the node module scope if they are top-level.
        # In vnccs_worldplay.py:
        # from worldplay.inference.pipeline_worldplay import WanPipeline -> heavy?
        # from worldplay.models.dits.arwan_w_action import WanTransformer3DModel -> heavy?
        pass

    def test_load_model(self):
        # Instantiate
        # Re-import to pick up new logic
        import importlib.util
        spec = importlib.util.spec_from_file_location("vnccs_worldplay", 
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nodes/vnccs_worldplay.py"))
        vnccs_worldplay = importlib.util.module_from_spec(spec)
        sys.modules["vnccs_worldplay"] = vnccs_worldplay
        spec.loader.exec_module(vnccs_worldplay)
        
        # Create mocks
        mock_hf_download = MagicMock()
        mock_snapshot = MagicMock()
        mock_pipe_load = MagicMock()
        mock_trans_load = MagicMock()
        
        # Patch the imported module's download functions
        vnccs_worldplay.hf_hub_download = mock_hf_download
        vnccs_worldplay.snapshot_download = mock_snapshot
        vnccs_worldplay.WanPipeline.from_pretrained = mock_pipe_load
        vnccs_worldplay.WanTransformer3DModel.from_pretrained = mock_trans_load

        # We need to mock os.path.exists inside the module to trigger downloads
        # Since os is imported in the module, we can patch it there
        
        # Create a mock for os that delegates everything but exists to real os? 
        # Easier: just patch vnccs_worldplay.os.path.exists
        
        # But os is likely imported as 'import os'.
        # So vnccs_worldplay.os is the module.
        
        # Patch os.path.exists globally but controlled
        original_exists = os.path.exists
        
        def side_effect(path):
            # Return False for our model paths to trigger download
            # print(f"DEBUG: Checking exists for {path}")
            if "worldplay" in str(path) or "models" in str(path):
                return False
            return True # Assume exists for imports etc
            
        with patch('os.path.exists', side_effect=side_effect):
             # Also enforce makedirs to not fail
             with patch('os.makedirs'):
                 node = vnccs_worldplay.VNCCS_LoadWorldPlay5B()
                 try:
                    node.load_model(precision="bf16")
                 except Exception:
                    pass

        # Verify calls
        # 1. Config json
        # 2. Safetensors
        # 3. Model pt
        self.assertTrue(mock_hf_download.call_count >= 3, "Specific files should be downloaded via hf_hub_download")
        
        # 4. VAE snapshot
        # 5. Text Enc snapshot
        self.assertTrue(mock_snapshot.call_count >= 2, "Snapshots for VAE/TextEnc should be called")

    def test_trajectory_modes(self):
        # Import dynamically again or reuse if class level import possible
        # For simplicity, re-importing logic here or assuming it's available from previous test load
        # Re-doing dynamic import to be safe
        import importlib.util
        spec = importlib.util.spec_from_file_location("vnccs_worldplay_traj", 
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nodes/vnccs_worldplay.py"))
        vnccs_worldplay = importlib.util.module_from_spec(spec)
        sys.modules["vnccs_worldplay_traj"] = vnccs_worldplay
        spec.loader.exec_module(vnccs_worldplay)
        
        traj_node = vnccs_worldplay.VNCCS_RoomCoverageTrajectory()
        
        # Test full_360
        res_360 = traj_node.generate_trajectory("full_360", 2, "")[0]
        self.assertIn("w-16, right-8", res_360)
        self.assertIn("up-8", res_360)
        
        # Test hemisphere
        res_hemi = traj_node.generate_trajectory("hemisphere", 3, "")[0]
        self.assertIn("down-8, right-48", res_hemi) # 3*16 = 48
        
        # Test orbital
        res_orb = traj_node.generate_trajectory("orbital", 2, "")[0]
        self.assertIn("d-16", res_orb)
        self.assertIn("left-4", res_orb)

    def test_optimizer_logic(self):
        # Import logic
        import importlib.util
        spec = importlib.util.spec_from_file_location("vnccs_worldplay_opt", 
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nodes/vnccs_worldplay.py"))
        vnccs_worldplay = importlib.util.module_from_spec(spec)
        sys.modules["vnccs_worldplay_opt"] = vnccs_worldplay
        spec.loader.exec_module(vnccs_worldplay)
        
        opt_node = vnccs_worldplay.VNCCS_VideoFrameOptimizer()
        
        # Create dummy images [B, H, W, C]
        # 10 frames.
        # 0, 1 identical. 2 different. 3 identical to 2.
        img0 = torch.zeros((10, 10, 3))
        img1 = torch.zeros((10, 10, 3)) # Duplicate of 0
        img2 = torch.ones((10, 10, 3))  # Different
        img3 = torch.ones((10, 10, 3)) # Duplicate of 2
        
        # Add slight noise to 3 to test threshold
        img3 += 0.001 
        
        batch = torch.stack([img0, img1, img2, img3]) # [4, 10, 10, 3]
        
        # Threshold 0.01. Diff(0,1)=0. Diff(1,2)=1.0. Diff(2,3)=0.001
        # Should keep 0, 2. (Indices 0, 2) -> 2 frames.
        # But min_frames default logic might kick in?
        
        # res is tuple (optimized, optimized)
        res_tuple = opt_node.optimize(batch, threshold=0.01, min_frames=1)
        self.assertEqual(len(res_tuple), 2, "Should return (video, image) tuple")
        res = res_tuple[0]
        
        # Expect 2 frames (0 and 2)
        # Note: logic keeps i if diff(i, last) > thresh.
        # i=1: diff(1,0)=0 < 0.01 -> Drop 1. Last=0.
        # i=2: diff(2,0)=1 > 0.01 -> Keep 2. Last=2.
        # i=3: diff(3,2)=0.001 < 0.01 -> Drop 3. Last=2.
        
        self.assertEqual(len(res), 2)
        self.assertTrue(torch.equal(res[0], img0))
        self.assertTrue(torch.equal(res[1], img2))
        
        # Test min frames enforcement
        # If we require 4 frames, should get all 4 back (interpolated indices)
        res_min = opt_node.optimize(batch, threshold=0.01, min_frames=4)[0]
        self.assertEqual(len(res_min), 4)

if __name__ == '__main__':
    unittest.main()
