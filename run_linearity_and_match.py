#!/usr/bin/env python3
"""Compare simplified VNCCS node vs original for a few weight settings and print diagnostics.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nodes'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'qweneditutils'))

import types
try:
    import torch
except Exception:
    print('torch needed for this test')
    raise

# Minimal mocks used by nodes
class MockComfyUtils:
    @staticmethod
    def common_upscale(samples, width, height, method, crop):
        return samples

class MockNodeHelpers:
    @staticmethod
    def conditioning_set_values(conditioning, values, append=False):
        new_conditioning = []
        for cond_tensor, cond_dict in conditioning:
            new = dict(cond_dict) if isinstance(cond_dict, dict) else {}
            if append:
                for k,v in values.items():
                    if k in new and isinstance(new[k], list):
                        new[k].extend(v if isinstance(v, list) else [v])
                    else:
                        new[k] = list(v) if isinstance(v, list) else [v]
            else:
                new.update(values)
            new_conditioning.append((cond_tensor, new))
        return new_conditioning

# Install mocks
import sys
comfy_mod = types.ModuleType('comfy')
comfy_mod.utils = MockComfyUtils()
sys.modules['comfy'] = comfy_mod
sys.modules['node_helpers'] = MockNodeHelpers()

# Import node implementations
from vnccs_qwen_encoder import VNCCS_QWEN_Encoder
from qweneditutils.nodes import TextEncodeQwenImageEditPlusAdvance_lrzjason as OrigNode

# Simple Mock CLIP and VAE to produce deterministic tensors
class MockCLIP:
    def __init__(self):
        pass
    def tokenize(self, text, images=None, llama_template=None):
        return {'input_ids': torch.randint(0,1000,(1,77))}
    def encode_from_tokens_scheduled(self, tokens):
        # produce a tensor depending on whether images provided in 'tokens' isn't accessible here,
        # but we'll return a deterministic tensor
        return [(torch.randn(1,77,128)*0.01 + 1.0, {})]

class MockVAE:
    def encode(self, image):
        return torch.randn(1,4,16,16)

# Create dummy images
def dummy_image():
    return torch.randn(1,3,64,64)

# runner
vn = VNCCS_QWEN_Encoder()
orig = OrigNode()
clip = MockCLIP()
vae = MockVAE()

cases = [
    ('all_ones', (1.0,1.0,1.0)),
    ('first_one_rest_zero', (1.0,0.0,0.0)),
    ('first_point_one', (0.1,0.0,0.0)),
    ('half_half', (0.5,0.5,0.0)),
]

for name, weights in cases:
    imgs = [dummy_image(), dummy_image(), None]
    print('\n=== CASE', name, 'weights=', weights, '===')
    torch.manual_seed(42)
    try:
        orig_out = orig.encode(clip=clip, prompt='hello', vae=vae, vl_resize_image1=None, vl_resize_image2=None, vl_resize_image3=None, not_resize_image1=imgs[0], not_resize_image2=imgs[1], not_resize_image3=imgs[2], target_size=1024, target_vl_size=392, upscale_method='lanczos', instruction='')
    except Exception as e:
        print('orig encode error', e)
        orig_out = None
    torch.manual_seed(42)
    try:
        vn_out = vn.encode(clip=clip, prompt='hello', vae=vae, image1=imgs[0], image2=imgs[1], image3=imgs[2], target_size=1024, target_vl_size=392, upscale_method='lanczos', instruction='', control_instruction='', weight1=weights[0], weight2=weights[1], weight3=weights[2])
    except Exception as e:
        print('vn encode error', e)
        vn_out = None

    def inspect(name, out, idx_full=0, idx_lat=1, idx_first=5, idx_weighted=6, idx_ref2=7):
        if out is None:
            print(name, 'None')
            return
        try:
            cond = out[idx_full]
            lat = out[idx_lat]
            # cond is list of (tensor, dict)
            t = cond[0][0]
            norms = float(torch.norm(t).item())
            refs = None
            try:
                refs = cond[0][1].get('reference_latents') if isinstance(cond[0][1], dict) else None
            except Exception:
                refs = None
            print(f"{name} cond_norm={norms:.6f} ref_count={len(refs) if refs is not None else 'None'}")
            # Also check conditioning_first_weighted if present
            if len(out) > idx_weighted:
                try:
                    cond_w = out[idx_weighted]
                    t_w = cond_w[0][0]
                    norms_w = float(torch.norm(t_w).item())
                    refs_w = cond_w[0][1].get('reference_latents') if isinstance(cond_w[0][1], dict) else None
                    print(f"{name} weighted_cond_norm={norms_w:.6f} ref_count={len(refs_w) if refs_w is not None else 'None'}")
                except Exception:
                    pass
            # Check conditioning_ref2_timing if present
            if len(out) > idx_ref2:
                try:
                    cond_r = out[idx_ref2]
                    t_r = cond_r[0][0]
                    norms_r = float(torch.norm(t_r).item())
                    refs_r = cond_r[0][1].get('reference_latents') if isinstance(cond_r[0][1], dict) else None
                    start_p = cond_r[0][1].get('start_percent') if isinstance(cond_r[0][1], dict) else None
                    end_p = cond_r[0][1].get('end_percent') if isinstance(cond_r[0][1], dict) else None
                    print(f"{name} ref2_cond_norm={norms_r:.6f} ref_count={len(refs_r) if refs_r is not None else 'None'} start={start_p} end={end_p}")
                except Exception:
                    pass
        except Exception as e:
            print('inspect error', e)

    inspect('ORIG', orig_out, idx_full=0, idx_lat=1, idx_first=8, idx_weighted=9, idx_ref2=10)  # adjust for orig
    inspect('VN', vn_out, idx_full=0, idx_lat=1, idx_first=5, idx_weighted=6, idx_ref2=7)

print('\nDone')
