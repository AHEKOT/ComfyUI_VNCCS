#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nodes'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'qweneditutils'))
import types
try:
    import torch
except Exception:
    raise

# mocks
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

import sys
comfy_mod = types.ModuleType('comfy')
comfy_mod.utils = MockComfyUtils()
sys.modules['comfy'] = comfy_mod
sys.modules['node_helpers'] = MockNodeHelpers()

from vnccs_qwen_encoder import VNCCS_QWEN_Encoder
from qweneditutils.nodes import TextEncodeQwenImageEditPlusAdvance_lrzjason as OrigNode

class MockCLIP:
    def tokenize(self, text, images=None, llama_template=None):
        return {'input_ids': torch.randint(0,1000,(1,77))}
    def encode_from_tokens_scheduled(self, tokens):
        return [(torch.randn(1,77,128)*0.01 + 1.0, {})]
class MockVAE:
    def encode(self, img):
        return torch.randn(1,4,16,16)

def dummy_image():
    return torch.randn(1,3,64,64)

clip = MockCLIP(); vae = MockVAE(); vn = VNCCS_QWEN_Encoder(); orig = OrigNode()

cases = [
    ('1_0_0',(1.0,0.0,0.0)),
    ('1_0.1_0',(1.0,0.1,0.0)),
]
for name, w in cases:
    print('\n---',name,'weights=',w)
    imgs = [dummy_image(), dummy_image(), None]
    out_orig = orig.encode(clip=clip, prompt='hello', vae=vae, vl_resize_image1=None, vl_resize_image2=None, vl_resize_image3=None, not_resize_image1=imgs[0], not_resize_image2=imgs[1], not_resize_image3=imgs[2], target_size=1024, target_vl_size=392, upscale_method='lanczos', instruction='')
    out_vn = vn.encode(clip=clip, prompt='hello', vae=vae, image1=imgs[0], image2=imgs[1], image3=imgs[2], target_size=1024, target_vl_size=392, upscale_method='lanczos', instruction='', weight1=w[0], weight2=w[1], weight3=w[2])
    def norm_of(cond):
        try:
            return float(torch.norm(cond[0][0]).item())
        except Exception:
            return None
    print('ORIG cond norm:', norm_of(out_orig[0]), 'ref_count:', len(out_orig[0][0][1].get('reference_latents') if isinstance(out_orig[0][0][1], dict) else []))
    print('VN full cond norm:', norm_of(out_vn[0]), 'ref_count:', len(out_vn[0][0][1].get('reference_latents') if isinstance(out_vn[0][0][1], dict) else []))
    print('VN with_first_ref norm:', norm_of(out_vn[5]))
    print('VN first_only norm:', norm_of(out_vn[6]))

print('\nDone')
