#!/usr/bin/env python3
"""SYNTHETIC landmark data, only to test the pipeline end to end.
Never use the numbers it produces as results. Real data comes from collect/."""
import argparse, json
from pathlib import Path
import numpy as np

BASE = {  # finger -> (mcp_x, mcp_y, base_angle_deg); wrist at (0.5, 0.8)
    "thumb": (0.43, 0.72, 40), "index": (0.45, 0.62, 10), "middle": (0.50, 0.60, 0),
    "ring": (0.55, 0.62, -8), "pinky": (0.59, 0.67, -18),
}
SEGS = [0.07, 0.05, 0.04]


def hand(curls, rng):
    pts = [[0.5, 0.8, 0.0]]
    for k, (name, (mx, my, ang)) in enumerate(BASE.items()):
        pts.append([mx, my, 0.0]) if name != "thumb" else None
        x, y, a = mx, my, np.deg2rad(ang)
        n = 3 if name != "thumb" else 4
        if name == "thumb":
            pts.append([x, y, 0.0])
        for i in range(n if name != "thumb" else 3):
            a += curls[k] * 0.9
            seg = SEGS[min(i, 2)]
            x -= seg * np.sin(a); y -= seg * np.cos(a)
            pts.append([x, y, float(rng_z(rng, curls[k], i))])
    return np.array(pts)[:21]


def rng_z(rng, c, i):
    return -0.02 * i * (1 - c) + rng.normal(0, 0.003)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signs", default="signs.json")
    ap.add_argument("--out", default="tests/synthetic_export.json")
    ap.add_argument("--signers", type=int, default=6)
    ap.add_argument("--per", type=int, default=40)
    a = ap.parse_args()
    cfg = json.load(open(a.signs))
    ids = [s["id"] for s in cfg["signs"]] + [cfg["none"]["id"]]
    rng = np.random.default_rng(7)
    protos = {i: rng.uniform(0, 1, 5) for i in ids[:-1]}
    samples = []
    for sg in range(a.signers):
        scale, rot, off = rng.uniform(0.8, 1.25), np.deg2rad(rng.uniform(-12, 12)), rng.normal(0, 0.03, 2)
        label = "Right" if sg % 3 else "Left"
        for sid in ids:
            for k in range(a.per):
                c = np.clip(protos[sid] + rng.normal(0, 0.06, 5), 0, 1) if sid != "none" else rng.uniform(0, 1, 5)
                p = hand(c, rng)
                p[:, :2] = (p[:, :2] - [0.5, 0.8]) * scale
                R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
                p[:, :2] = p[:, :2] @ R.T + [0.5, 0.8] + off + rng.normal(0, 0.004, (21, 2))
                if label == "Left":
                    p[:, 0] = 1 - p[:, 0]
                samples.append({"sign": sid, "signer_id": f"SYN{sg+1:02d}", "signer_type": "synthetic",
                                "consent": True, "hand": label, "hand_score": 0.99,
                                "landmarks": np.round(p, 5).tolist(), "aspect": 1.3333,
                                "device": "synthetic", "lighting": "normal", "distance": "mid",
                                "take_id": f"t{sg}-{sid}-{k//10}", "created": "2026-11-14T00:00:00Z"})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"format": "lingora-landmarks-v1", "synthetic": True, "samples": samples}, open(a.out, "w"))
    print(f"wrote {len(samples)} synthetic samples -> {a.out}")


if __name__ == "__main__":
    main()
