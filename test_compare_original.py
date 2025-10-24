#!/usr/bin/env python3
"""
Compare VNCCS_QWEN_Encoder (nodes/vnccs_qwen_encoder.py) vs original
TextEncodeQwenImageEditPlusAdvance (qweneditutils/nodes_exp.py) behavior.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
# ensure nodes and qweneditutils are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nodes'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'qweneditutils'))

try:
    import torch
except Exception:
    print("[test_compare_original] torch not available — skipping.")
    sys.exit(0)

# Reuse mocks from test_vnccs_weights
class MockComfyUtils:
    @staticmethod
    def common_upscale(samples, width, height, method, crop):
        # Lightweight passthrough: avoid expensive interpolation in tests
        return samples

class MockNodeHelpers:
    @staticmethod
    def conditioning_set_values(conditioning, values, append=False):
        new_conditioning = []
        for cond_tensor, cond_dict in conditioning:
            new_dict = cond_dict.copy()
            if append:
                for key, val in values.items():
                    if key in new_dict and isinstance(new_dict[key], list):
                        new_dict[key].extend(val)
                    else:
                        new_dict[key] = list(val) if isinstance(val, list) else [val]
            else:
                new_dict.update(values)
            new_conditioning.append((cond_tensor, new_dict))
        return new_conditioning

class MockVAE:
    def encode(self, image_proc):
        # return a small latent to keep memory low
        return torch.randn(1, 4, 16, 16)

class MockCLIP:
    def tokenize(self, text, images=None, llama_template=None):
        return {"input_ids": torch.randint(0, 1000, (1, 77))}
    def encode_from_tokens_scheduled(self, tokens):
        # smaller conditioning tensor for lightweight tests
        return [(torch.randn(1, 77, 128), {})]

# Install mocks into sys.modules for imports that expect comfy/node_helpers
import types
comfy_mod = types.ModuleType('comfy')
comfy_mod.utils = MockComfyUtils()
sys.modules['comfy'] = comfy_mod
sys.modules['node_helpers'] = MockNodeHelpers()

# Import nodes (after installing mocks)
from vnccs_qwen_encoder import VNCCS_QWEN_Encoder
from qweneditutils.nodes import TextEncodeQwenImageEditPlusAdvance_lrzjason

# Create dummy images
def create_dummy_image():
    # Much smaller image to avoid heavy ops
    return torch.randn(1, 3, 64, 64)

# Runner
def run_compare():
    vn_node = VNCCS_QWEN_Encoder()
    orig_node = TextEncodeQwenImageEditPlusAdvance_lrzjason()
    clip = MockCLIP()
    vae = MockVAE()

    # Build a list of image-presence cases
    image_cases = [
        ("single_image", [create_dummy_image(), None, None]),
        ("two_images", [create_dummy_image(), create_dummy_image(), None]),
        ("three_images", [create_dummy_image(), create_dummy_image(), create_dummy_image()]),
        ("no_vae_images", [None, None, None]),
    ]

    weight_sets = [
        (1.0, 1.0, 1.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.0001, 0.0),
        (0.5, 0.5, 0.0),
        (0.2, 0.8, 0.0),
        (2.0, 1.0, 0.0),
    ]

    mapping_modes = ["exponential", "linear"]
    sensitivities = ["normal", "fine", "ultrafine"]
    weights_affect_latent_opts = [True, False]

    # We'll collect a list of mismatches to show at the end
    mismatches = []

    for img_case_name, imgs in image_cases:
        for weights in weight_sets:
            for mapping in mapping_modes:
                for sens in sensitivities:
                    for weights_affect in weights_affect_latent_opts:
                        print(f"\n=== TEST: {img_case_name} weights={weights} mapping={mapping} sens={sens} latents_affect={weights_affect} ===")
                        # Prepare VAE presence: allow turning off VAE when all images are None in test case
                        vae_used = vae if any(img is not None for img in imgs) else None

                        # Run original
                        try:
                            torch.manual_seed(12345)
                            orig_out = orig_node.encode(clip=clip, prompt='hello', vae=vae_used, vl_resize_image1=None, vl_resize_image2=None, vl_resize_image3=None, not_resize_image1=imgs[0], not_resize_image2=imgs[1], not_resize_image3=imgs[2], target_size=1024, target_vl_size=392, upscale_method='lanczos', instruction='')
                        except Exception as e:
                            print(f"[original] ERROR: {e}")
                            orig_out = None

                        # Run VNCCS
                        try:
                            torch.manual_seed(12345)
                            vn_out = vn_node.encode(clip=clip, prompt='hello', vae=vae_used, image1=imgs[0], image2=imgs[1], image3=imgs[2], target_size=1024, target_vl_size=392, upscale_method='lanczos', instruction='', weight1=weights[0], weight2=weights[1], weight3=weights[2])
                        except Exception as e:
                            print(f"[vnccs] ERROR: {e}")
                            vn_out = None

        # Helper: compare two conditioning objects (conditioning lists of (tensor, dict))
        def compare_conditionings(name, cond_full, cond_first):
            if cond_full is None:
                print(f"{name}: conditioning_full_ref is None")
                return
            if cond_first is None:
                print(f"{name}: conditioning_with_first_ref is None")
                return
            try:
                tf = cond_full[0][0]
                t1 = cond_first[0][0]
                norm_full = torch.norm(tf).item()
                norm_first = torch.norm(t1).item()
                diff = tf - t1
                max_abs = float(torch.max(torch.abs(diff)).item())
                mean_abs = float(torch.mean(torch.abs(diff)).item())
                allclose = False
                try:
                    allclose = torch.allclose(tf, t1)
                except Exception:
                    allclose = abs(norm_full - norm_first) < 1e-6
                print(f"{name} conditioning_full_ref norm={norm_full:.6f}, conditioning_with_first_ref norm={norm_first:.6f}, max_abs_diff={max_abs:.6g}, mean_abs_diff={mean_abs:.6g}, allclose={allclose}")
            except Exception as e:
                print(f"{name} conditioning compare error: {e}")
            # inspect reference_latents metadata
            try:
                ref_full = cond_full[0][1].get('reference_latents') if isinstance(cond_full[0][1], dict) else None
                ref_first = cond_first[0][1].get('reference_latents') if isinstance(cond_first[0][1], dict) else None
                cnt_full = len(ref_full) if ref_full is not None else 0
                cnt_first = len(ref_first) if ref_first is not None else 0
                print(f"{name} reference_latents counts: full={cnt_full}, first={cnt_first}")
                if ref_full is not None:
                    for i, r in enumerate(ref_full):
                        try:
                            print(f" {name} ref_full[{i}] shape: {tuple(r.shape) if hasattr(r, 'shape') else type(r)}")
                        except Exception:
                            print(f" {name} ref_full[{i}] shape: <error>")
                if ref_first is not None:
                    for i, r in enumerate(ref_first):
                        try:
                            print(f" {name} ref_first[{i}] shape: {tuple(r.shape) if hasattr(r, 'shape') else type(r)}")
                        except Exception:
                            print(f" {name} ref_first[{i}] shape: <error>")
            except Exception as e:
                print(f"{name} ref inspect error: {e}")

        # For original node: conditioning is at out[0]; conditioning_with_first_ref is returned later (index 8 when present)
        if orig_out is not None:
            try:
                orig_cond_full = orig_out[0]
                # original node returns conditioning_with_first_ref near the end (index -2 or index 8)
                if len(orig_out) >= 9:
                    orig_cond_first = orig_out[8]
                else:
                    orig_cond_first = orig_out[0]
            except Exception:
                orig_cond_full = None
                orig_cond_first = None
        else:
            orig_cond_full = None
            orig_cond_first = None

        # For VNCCS node: conditioning_full_ref is out[0], conditioning_with_first_ref is out[5]
        if vn_out is not None:
            try:
                vn_cond_full = vn_out[0]
                vn_cond_first = vn_out[5]
            except Exception:
                vn_cond_full = None
                vn_cond_first = None
        else:
            vn_cond_full = None
            vn_cond_first = None

        # Compare conditioning objects
        try:
            compare_conditionings('original', orig_cond_full, orig_cond_first)
        except Exception:
            print('[original] conditioning compare failed')
        try:
            compare_conditionings('vnccs', vn_cond_full, vn_cond_first)
        except Exception:
            print('[vnccs] conditioning compare failed')

        # Compare latents (samples)
        try:
            orig_lat = orig_out[1]['samples'] if orig_out is not None and isinstance(orig_out[1], dict) else (orig_out[1] if orig_out is not None else None)
        except Exception:
            orig_lat = None
        try:
            vn_lat = vn_out[1]['samples'] if vn_out is not None and isinstance(vn_out[1], dict) else (vn_out[1] if vn_out is not None else None)
        except Exception:
            vn_lat = None

        lat_mismatch = False
        try:
            if orig_lat is None and vn_lat is None:
                lat_mismatch = False
            elif (orig_lat is None) != (vn_lat is None):
                lat_mismatch = True
            else:
                # both present: compare shapes and allclose
                s1 = tuple(orig_lat.shape) if hasattr(orig_lat, 'shape') else None
                s2 = tuple(vn_lat.shape) if hasattr(vn_lat, 'shape') else None
                if s1 != s2:
                    lat_mismatch = True
                else:
                    allc = False
                    try:
                        allc = torch.allclose(orig_lat, vn_lat)
                    except Exception:
                        allc = float(torch.norm(orig_lat - vn_lat).abs().max()) < 1e-6
                    lat_mismatch = not allc
        except Exception:
            lat_mismatch = True

        # Collect mismatches
        if lat_mismatch:
            mismatches.append((img_case_name, weights, mapping, sens, weights_affect))

    # Summary
    print('\n\n=== MISMATCH SUMMARY ===')
    if len(mismatches) == 0:
        print('No latent mismatches detected across tested combos.')
    else:
        for m in mismatches:
            print('Mismatch at', m)

if __name__ == '__main__':
    run_compare()
