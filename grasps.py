"""
grasps.py — hand posture data.

IMPORTANT, and say this on the slide: the postures generated here are a
STAND-IN so the pipeline can be built and tested before you have real data.
They are not evidence of anything about real hands. The Saturday job is to
replace them with your own recordings.

Joint set matches Santello et al. so the comparison is meaningful: 15 angles,
being MCP and PIP flexion for five digits, four inter-finger abductions, and
thumb opposition.

On circularity, because a judge may ask and you should have the answer ready:
each grasp type below is defined by its FUNCTIONAL description, one joint at a
time. No synergy basis is imposed anywhere in this file. If PCA then finds that
two components explain most of the variance, that structure came from the fact
that real grasps constrain joints together, which is exactly the phenomenon
Santello reported. Generating data from a 2D basis and then "discovering" 2
dimensions would prove nothing, and this deliberately does not do that.
"""

import numpy as np

JOINTS = [
    "thumb_MCP", "thumb_IP", "index_MCP", "index_PIP", "middle_MCP",
    "middle_PIP", "ring_MCP", "ring_PIP", "pinky_MCP", "pinky_PIP",
    "abd_thumb_index", "abd_index_middle", "abd_middle_ring",
    "abd_ring_pinky", "thumb_opposition",
]

# Degrees. Written per joint from the functional description of each grasp.
GRASP_TYPES = {
    "power_sphere":   [35, 40, 55, 70, 60, 75, 58, 72, 55, 68, 25, 12, 10,  9, 45],
    "power_cylinder": [40, 50, 70, 85, 75, 90, 72, 88, 68, 85, 18,  6,  5,  5, 55],
    "precision_pinch":[45, 55, 45, 25, 15, 20, 12, 18, 10, 15, 30,  8,  6,  6, 70],
    "tripod":         [42, 50, 48, 35, 45, 32, 15, 20, 12, 16, 28, 10,  8,  6, 65],
    "lateral_key":    [30, 20, 75, 80, 72, 82, 70, 80, 68, 78,  8,  5,  4,  4, 20],
    "hook":           [10,  8, 80, 95, 85, 98, 82, 95, 78, 92,  5,  4,  4,  4,  5],
    "open_palm":      [ 8,  5,  6,  8,  5,  6,  5,  7,  5,  8, 20, 15, 12, 12, 15],
    "index_point":    [35, 40,  8, 10, 80, 92, 78, 90, 75, 88, 15,  8,  5,  5, 40],
    "large_diameter": [38, 45, 50, 62, 55, 68, 52, 65, 50, 62, 22, 10,  8,  7, 50],
    "small_pinch":    [48, 60, 52, 40, 20, 25, 14, 18, 11, 14, 32,  9,  6,  5, 75],
    "flat_grip":      [20, 15, 25, 30, 28, 32, 26, 30, 24, 28, 12,  6,  5,  5, 25],
    "fist":           [45, 55, 85, 98, 88, 99, 85, 97, 82, 95,  6,  4,  3,  3, 60],
}


def sample_postures(n_per_grasp=12, jitter=6.0, seed=0):
    """
    Return (poses, labels). Jitter is per-joint independent Gaussian noise,
    which if anything works AGAINST finding low-dimensional structure.
    """
    rng = np.random.default_rng(seed)
    poses, labels = [], []
    for name, base in GRASP_TYPES.items():
        b = np.array(base, dtype=float)
        for _ in range(n_per_grasp):
            poses.append(b + rng.normal(0, jitter, b.shape))
            labels.append(name)
    P = np.array(poses)
    order = rng.permutation(len(P))
    return P[order], [labels[i] for i in order]


def train_test_split(poses, frac=0.7, seed=1):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(poses))
    cut = int(frac * len(poses))
    return poses[idx[:cut]], poses[idx[cut:]]
