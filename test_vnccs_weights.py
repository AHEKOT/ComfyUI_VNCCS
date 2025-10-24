#!/usr/bin/env python3
"""
Minimal test script for VNCCS_QWEN_Encoder weight simulation.
Mocks ComfyUI dependencies to validate weight mapping and delta combination logic.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nodes'))

try:
    import torch
except Exception:
    print("[test_vnccs_weights] torch not available — skipping simulation. Install torch to run this test.")
    sys.exit(0)
try:
    import numpy as np
except Exception:
    import math as np  # fallback minimal
try:
    from PIL import Image
except Exception:
    Image = None
import math

# Mock ComfyUI modules
class MockComfyUtils:
    @staticmethod
    def common_upscale(samples, width, height, method, crop):
        # Mock upscale: just resize the tensor
        return torch.nn.functional.interpolate(samples, size=(height, width), mode='bilinear', align_corners=False)

class MockNodeHelpers:
    @staticmethod
    def conditioning_set_values(conditioning, values, append=False):
        # Mock: conditioning is list of (tensor, dict)
        new_conditioning = []
        for cond_tensor, cond_dict in conditioning:
            new_dict = cond_dict.copy()
            if append:
                for key, val in values.items():
                    if key in new_dict:
                        new_dict[key].extend(val)
                    else:
                        new_dict[key] = val
            else:
                new_dict.update(values)
            new_conditioning.append((cond_tensor, new_dict))
        return new_conditioning

class MockVAE:
    def encode(self, image_proc):
        # Mock: return random latent
        return torch.randn(1, 4, 64, 64)

class MockCLIP:
    def tokenize(self, text, images=None, llama_template=None):
        # Mock tokens
        return {"input_ids": torch.randint(0, 1000, (1, 77))}

    def encode_from_tokens_scheduled(self, tokens):
        # Mock conditioning: random tensor
        return [(torch.randn(1, 77, 1280), {})]

# Monkey patch and import the node safely
import sys
import types
sys.path.insert(0, 'nodes')
try:
    comfy_mock = types.ModuleType('comfy')
    comfy_mock.utils = MockComfyUtils()
    sys.modules['comfy'] = comfy_mock
    sys.modules['node_helpers'] = MockNodeHelpers()
    import vnccs_qwen_encoder
except Exception as e:
    print(f"[test_vnccs_weights] Could not import node module: {e}\nMake sure you're running this from the repository root with the 'nodes' folder present and that ComfyUI dependencies are installed.")
    sys.exit(0)

# Create dummy images
def create_dummy_image():
    return torch.randn(1, 3, 512, 512)  # CHW

# Test function
def test_weights():
    node = vnccs_qwen_encoder.VNCCS_QWEN_Encoder()
    clip = MockCLIP()
    vae = MockVAE()

    # Test cases
    test_cases = [
        {"name": "single_image_weight_1", "images": [create_dummy_image(), None, None], "weights": [1.0, 1.0, 1.0]},
        {"name": "two_images_weights_1_0", "images": [create_dummy_image(), create_dummy_image(), None], "weights": [1.0, 0.0, 1.0]},
        {"name": "two_images_weights_0.5_0.5", "images": [create_dummy_image(), create_dummy_image(), None], "weights": [0.5, 0.5, 1.0]},
        {"name": "three_images_weights_1_1_1", "images": [create_dummy_image(), create_dummy_image(), create_dummy_image()], "weights": [1.0, 1.0, 1.0]},
    ]

    mapping_modes = ["exponential", "linear"]
    sensitivities = ["normal", "fine", "ultrafine"]
    for case in test_cases:
        print(f"\nCase: {case['name']}")
        for mapping in mapping_modes:
            for sens in sensitivities:
                try:
                    result = node.encode(
                        clip=clip,
                        prompt="test prompt",
                        vae=vae,
                        image1=case['images'][0],
                        image2=case['images'][1],
                        image3=case['images'][2],
                        weight1=case['weights'][0],
                        weight2=case['weights'][1],
                        weight3=case['weights'][2],
                        conditioning_sensitivity=sens,
                        weight_mapping=mapping,
                        debug_logging=False
                    )
                    conditioning_full_ref, latent, img1, img2, img3, conditioning_with_first_ref = result
                    norm_full = torch.norm(conditioning_full_ref[0][0]).item()
                    norm_first = torch.norm(conditioning_with_first_ref[0][0]).item()
                    diff = abs(norm_full - norm_first)
                    print(f" mapping={mapping:11s} sens={sens:8s} -> full_norm={norm_full:.6f} first_norm={norm_first:.6f} diff={diff:.6f}")
                except Exception as e:
                    print(f" mapping={mapping:11s} sens={sens:8s} -> ERROR: {e}")

if __name__ == "__main__":
    test_weights()