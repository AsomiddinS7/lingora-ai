"""Feature extraction v1. MUST stay identical to features.js (parity is tested).

Input : 21 hand landmarks (x, y, z) from MediaPipe, the MediaPipe handedness
        label ("Left" or "Right"), and the video aspect ratio (width / height).
Output: 78 numbers: 63 normalized coordinates + 15 distances.
"""
import numpy as np

FEATURE_VERSION = "v1"
FEATURE_DIM = 78
TIPS = [4, 8, 12, 16, 20]
PAIRS = [(TIPS[a], TIPS[b]) for a in range(5) for b in range(a + 1, 5)]  # 10 pairs


def features(lm, hand="Right", aspect=4 / 3):
    p = np.asarray(lm, dtype=np.float64).reshape(21, 3).copy()
    p[:, 0] *= aspect              # make x and y the same scale
    p -= p[0]                      # wrist at the origin
    if hand == "Left":             # one canonical hand shape, whatever the label
        p[:, 0] *= -1
    s = np.linalg.norm(p[9])       # palm size: wrist to middle-finger base
    p /= s if s > 1e-6 else 1.0
    d = [np.linalg.norm(p[t]) for t in TIPS]
    d += [np.linalg.norm(p[i] - p[j]) for i, j in PAIRS]
    return np.concatenate([p.ravel(), np.array(d)]).astype(np.float32)


def augment_landmarks(lm, aspect, rng, rot_deg=(6.0, 6.0, 10.0), jitter=0.02):
    """Random small 3D rotation + jitter. Returns landmarks in the ORIGINAL
    normalized coordinates, so features(..., aspect) can be applied unchanged."""
    p = np.asarray(lm, dtype=np.float64).reshape(21, 3).copy()
    p[:, 0] *= aspect
    p -= p[0]
    ax, ay, az = [np.deg2rad(np.clip(rng.normal(0, r / 2), -r, r)) for r in rot_deg]
    cx, sx, cy, sy, cz, sz = np.cos(ax), np.sin(ax), np.cos(ay), np.sin(ay), np.cos(az), np.sin(az)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    p = p @ (rz @ ry @ rx).T
    s = np.linalg.norm(p[9])
    p += rng.normal(0, jitter * (s if s > 1e-6 else 1.0), p.shape)
    p[:, 0] /= aspect
    return p
