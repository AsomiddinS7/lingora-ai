// Feature extraction v1. MUST stay identical to features.py (parity is tested).
export const FEATURE_VERSION = "v1";
export const FEATURE_DIM = 78;
const TIPS = [4, 8, 12, 16, 20];
const PAIRS = [];
for (let a = 0; a < 5; a++) for (let b = a + 1; b < 5; b++) PAIRS.push([TIPS[a], TIPS[b]]);

const norm3 = (u, v) => Math.hypot(u[0] - v[0], u[1] - v[1], u[2] - v[2]);

/** lm: array of 21 [x, y, z]; hand: "Left" | "Right"; aspect: video width / height */
export function features(lm, hand = "Right", aspect = 4 / 3) {
  const p = lm.map((q) => [q[0] * aspect, q[1], q[2]]);
  const w = p[0].slice();
  for (const q of p) { q[0] -= w[0]; q[1] -= w[1]; q[2] -= w[2]; }
  if (hand === "Left") for (const q of p) q[0] *= -1;
  let s = Math.hypot(p[9][0], p[9][1], p[9][2]);
  if (!(s > 1e-6)) s = 1;
  for (const q of p) { q[0] /= s; q[1] /= s; q[2] /= s; }
  const out = new Float32Array(FEATURE_DIM);
  let k = 0;
  for (const q of p) { out[k++] = q[0]; out[k++] = q[1]; out[k++] = q[2]; }
  const o = [0, 0, 0];
  for (const t of TIPS) out[k++] = norm3(p[t], o);
  for (const [i, j] of PAIRS) out[k++] = norm3(p[i], p[j]);
  return out;
}
