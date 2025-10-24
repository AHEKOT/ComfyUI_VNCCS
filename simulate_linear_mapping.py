"""Small simulator to validate linear weight mapping and its effect on conditioning/latents.
This script does not touch the node; it's just pure python math and small mock tensors.
"""
import math
import random

import numpy as np


def map_linear(w: float) -> float:
    try:
        w = float(w)
    except Exception:
        return 0.0
    if w <= 0.0:
        return 0.0
    if w <= 1.0:
        return w
    return 1.0 + 0.3 * (w - 1.0)


def simulate(weights, cond_norms=(4190.0, 4037.0, 1000.0)):
    mapped = [map_linear(w) for w in weights]
    # inclusion decision: include if mapped > 0
    append_idxs = [i for i,m in enumerate(mapped) if m > 0.0]

    # Mock cond_single tensors by their norms; create vectors with given norm
    cond_vectors = []
    for n in cond_norms:
        # create a vector with length 1024 scaled to have L2 norm ~= n
        v = np.ones(1024, dtype=np.float32)
        cur_norm = np.linalg.norm(v)
        v = v * (n / cur_norm)
        cond_vectors.append(v)

    # compute weighted average using mapped weights
    weighted_sum = None
    total_w = 0.0
    for i in append_idxs:
        w = mapped[i]
        v = cond_vectors[i]
        if weighted_sum is None:
            weighted_sum = v * w
        else:
            weighted_sum = weighted_sum + v * w
        total_w += w
    if weighted_sum is not None and total_w > 0.0:
        combined = weighted_sum / total_w
        comb_norm = np.linalg.norm(combined)
    else:
        combined = cond_vectors[0]
        comb_norm = np.linalg.norm(combined)

    print("weights:", weights)
    print("mapped:", mapped)
    print("append_idxs:", append_idxs)
    print("combined_norm: {:.6f}".format(comb_norm))
    # Show relative change vs only-first
    first_norm = np.linalg.norm(cond_vectors[0])
    diff = (comb_norm - first_norm) / (first_norm if first_norm != 0 else 1)
    print("relative change vs only-first: {:.6%}".format(diff))
    print("---")


if __name__ == '__main__':
    cases = [
        (1.0, 0.0001, 0.0),
        (0.1, 0.0, 0.0),
        (0.0001, 0.0001, 0.0),
        (0.5, 0.5, 0.0),
    ]
    for c in cases:
        simulate(c)
