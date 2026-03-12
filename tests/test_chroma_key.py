
import unittest
import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes.vnccs_utils import VNCCSChromaKey

class TestVNCCSChromaKey(unittest.TestCase):
    def setUp(self):
        self.node = VNCCSChromaKey()

    def test_pure_green_background(self):
        # Create a 100x100 image with green background and a black square in middle
        image = torch.zeros((100, 100, 3))
        image[:, :, 1] = 1.0 # Pure green [0, 1, 0]
        
        # Add a 50x50 black square in the middle
        # Middle is 25:75, 25:75
        image[25:75, 25:75, :] = 0.5 # Gray square
        
        # Test detection (internal call to chroma_key_single)
        # We need to test the color detection logic specifically
        # Currently it's inline, so we test the whole function
        
        # image input to chroma_key is [B, H, W, C]
        image_batch = image.unsqueeze(0)
        result_img_batch, = self.node.chroma_key(image_batch, tolerance=0.1, despill_strength=0.0, despill_kernel_size=3, despill_color="black")
        
        result_img = result_img_batch[0]
        
        # Alpha channel is the 4th channel
        alpha = result_img[:, :, 3]
        
        # Background should be transparent (alpha=0)
        # Middle square should be opaque (alpha=1)
        
        # Corner pixel (background)
        self.assertLess(alpha[0, 0], 0.1)
        # Center pixel (subject)
        self.assertGreater(alpha[50, 50], 0.9)

    def test_noisy_corners(self):
        # Create a 100x100 green image
        image = torch.zeros((100, 100, 3))
        image[:, :, 1] = 1.0 
        
        # Corrupt Top-Left corner with noise/other color
        image[0:10, 0:10, :] = torch.rand((10, 10, 3))
        
        # Center subject
        image[40:60, 40:60, :] = 0.5
        
        image_batch = image.unsqueeze(0)
        # If detection is robust, it should ignore the nose and still pick green.
        # Current logic (median of all borders) is already somewhat robust.
        
        result_img_batch, = self.node.chroma_key(image_batch, tolerance=0.1, despill_strength=0.0, despill_kernel_size=3, despill_color="black")
        result_img = result_img_batch[0]
        alpha = result_img[:, :, 3]
        
        # Bottom-Right corner should be transparent if green was detected
        self.assertLess(alpha[99, 99], 0.1)

if __name__ == '__main__':
    unittest.main()
