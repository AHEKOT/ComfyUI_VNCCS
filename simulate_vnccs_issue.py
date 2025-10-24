#!/usr/bin/env python3
"""
Simulation to reproduce VNCCS_QWEN_Encoder branch where weight1=1, weight2=0 and two images,
but conditioning_with_first_ref != conditioning_full_ref.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nodes'))
import torch
import types

# Import node
from vnccs_qwen_encoder import VNCCS_QWEN_Encoder

# Helper to build tensors with given L2 norm
def tensor_with_norm(shape, target_norm):
    t = torch.ones(shape, dtype=torch.float32)
    cur = torch.norm(t).item()
    if cur == 0:
        return t
    scale = target_norm / cur
    return t * scale

class MockCLIP:
    def __init__(self):
        # prepare tensors to be returned
        # conditioning_base: returned when tokenize(images=[]) used
        self.cond_base = [(tensor_with_norm((1,77,128), 1767.017822), {})]
        # cond_single for src0
        self.cond_src0 = [(tensor_with_norm((1,77,128), 4190.173340), {})]
        # cond_single for src1
        self.cond_src1 = [(tensor_with_norm((1,77,128), 4037.030273), {})]
        # tokens for multi-image call should normally return some combined conditioning
        self.combined = [(tensor_with_norm((1,77,128), 4900.427967), {})]
    def tokenize(self, text, images=None, llama_template=None):
        # the node passes images list; we don't need to return realistic tokens
        return {'input_ids': torch.randint(0,1000,(1,77))}
    def encode_from_tokens_scheduled(self, tokens):
        # decide which to return based on some global tracking of last call
        # We'll use a simple heuristic: if tokens looks like a dict from tokenize, inspect length of images via a side channel
        # But here we'll rely on caller to set a flag on this object before each call
        caller = getattr(self, '_next', 'combined')
        self._next = 'combined'
        if caller == 'base':
            return self.cond_base
        if caller == 'src0':
            return self.cond_src0
        if caller == 'src1':
            return self.cond_src1
        return self.combined

# Create mocks and set clip to produce desired sequence
clip = MockCLIP()
vae = types.SimpleNamespace(encode=lambda img: torch.randn(1,4,16,16))

# Prepare two dummy images (simple tensors)
img1 = torch.randn(1,3,64,64)
img2 = torch.randn(1,3,64,64)

node = VNCCS_QWEN_Encoder()

# We'll emulate the calls order in node.encode():
# 1) clip.tokenize(image_prompt + prompt, images=valid_vl_images) -> combined
# 2) tokens_base = clip.tokenize(prompt, images=[]) -> base
# 3) per-image tokens_single calls -> src0, src1

# To influence encode_from_tokens_scheduled returns, set clip._next before each external call via monkeypatching tokenize to set it
orig_tokenize = clip.tokenize

def tokenize_and_choose(which):
    def tok(text, images=None, llama_template=None):
        clip._next = which
        return orig_tokenize(text, images=images, llama_template=llama_template)
    return tok

# Now craft a wrapper around node.encode to force our sequence
# We'll monkeypatch clip.tokenize during the call by replacing methods dynamically

# Case: two images, weights (1,0,0)
print('\n--- Simulating two images, weights (1,0,0) ---')
# Set up tokenization sequence by swapping tokenize at specific times inside encode
# We'll run encode but intercept calls by temporarily swapping clip.tokenize

# 1) For multi-image call (first), return combined
clip.tokenize = tokenize_and_choose('combined')
# 2) The node will then call clip.encode_from_tokens_scheduled -> will return combined
# execute first part by calling encode but we need to ensure subsequent calls get other flags.
# To orchestrate properly, we'll create a small wrapper that swaps clip.tokenize appropriately at runtime

# We'll patch the node's clip param to our clip and then run encode while swapping tokenize at points using a generator.

# Simpler approach: run encode normally but let encode() make calls; before calling encode, set clip._next default to 'combined', and inside encode we want the second call to be 'base' and then per-image 'src0' and 'src1'.
# We'll intercept clip.tokenize to pop from a prepared list of which-return-values.

sequence = ['combined', 'base', 'src0', 'src1']

def seq_tokenize(text, images=None, llama_template=None):
    which = sequence.pop(0) if sequence else 'combined'
    clip._next = which
    return orig_tokenize(text, images=images, llama_template=llama_template)

clip.tokenize = seq_tokenize

# Run encode
out = node.encode(clip=clip, prompt='test', vae=vae, image1=img1, image2=img2, image3=None, weight1=1.0, weight2=0.0001, weight3=0.0)

# Print debug outputs captured in node.encode (it prints to stdout). Also print returned conditioning norms and ref counts
cond_full = out[0]
cond_first = out[5]
print('\n--- Post-run inspect ---')
try:
    tf = cond_full[0][0]
    t1 = cond_first[0][0]
    print('conditioning_full norm:', float(torch.norm(tf).item()))
    print('conditioning_first norm:', float(torch.norm(t1).item()))
    print('allclose:', torch.allclose(tf, t1))
except Exception as e:
    print('inspect error', e)

try:
    ref_full = cond_full[0][1].get('reference_latents') if isinstance(cond_full[0][1], dict) else None
    ref_first = cond_first[0][1].get('reference_latents') if isinstance(cond_first[0][1], dict) else None
    print('ref_full count:', len(ref_full) if ref_full is not None else 0)
    print('ref_first count:', len(ref_first) if ref_first is not None else 0)
except Exception as e:
    print('ref inspect error', e)

print('\nSimulation finished')
