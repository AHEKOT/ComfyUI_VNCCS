"""Simulator for delta-based linear mapping:
combined = conditioning_base + sum_i w_i * (cond_i - conditioning_base)
This models the simple behavior you asked for: weights are linear factors on the delta
from the base conditioning.
"""
import numpy as np


def simulate(weights, cond_norms=(1767.0, 4190.0, 4037.0)):
    weights = [float(w) for w in weights]
    # Build mock vectors for base and per-image cond_single
    size = 1024
    base = np.ones(size, dtype=np.float32) * (cond_norms[0] / np.linalg.norm(np.ones(size)))
    conds = []
    for n in [cond_norms[1], cond_norms[2], 1000.0]:
        v = np.ones(size, dtype=np.float32)
        v = v * (n / np.linalg.norm(v))
        conds.append(v)

    # compute combined = base + sum_i w_i * (cond_i - base)
    combined = np.array(base, copy=True)
    for i, w in enumerate(weights[:2]):
        combined += w * (conds[i] - base)

    comb_norm = np.linalg.norm(combined)
    base_norm = np.linalg.norm(base)

    print("weights:", weights)
    print(f"base_norm: {base_norm:.6f}, combined_norm: {comb_norm:.6f}")
    rel = (comb_norm - base_norm) / (base_norm if base_norm!=0 else 1)
    print(f"relative change vs base: {rel:.6%}")
    print('---')


if __name__ == '__main__':
    cases = [
        (1.0, 0.0),   # full first image
        (0.1, 0.0),   # 10% first image
        (0.0001, 0.0),# very tiny
        (0.5, 0.5),   # two images equal
    ]
    for c in cases:
        simulate(c)
