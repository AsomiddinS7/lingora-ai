import assert from "node:assert/strict";
import { SignerLock, filterHands, personFromPose, focusedHand } from "../shared/signer_lock.js";

// A person at anchor (x, y) with shoulder width sw. Wrists hang by default.
const P = (x, y, sw, wrists) => ({ anchor: [x, y], sw, nose: [x, y - sw], wrists: wrists ?? [[x + sw, y + 2 * sw], [x - sw, y + 2 * sw]] });
const run = (lock, frames, dt = 66) => frames.map((f, i) => lock.update(f, i * dt));

// 1. auto mode picks the closest (largest) person, flags a close call
let l = new SignerLock();
let r = l.update([P(0.3, 0.4, 0.12), P(0.8, 0.4, 0.30)], 0);
assert.equal(r.index, 1); assert.equal(r.state, "locked"); assert.equal(r.ambiguous, false);
l = new SignerLock();
assert.equal(l.update([P(0.3, 0.4, 0.29), P(0.8, 0.4, 0.30)], 0).ambiguous, true);

// 2. a bigger person walking in does not steal the lock
l = new SignerLock();
l.update([P(0.5, 0.4, 0.2)], 0);
let out = run(l, Array.from({ length: 30 }, (_, i) => [P(0.5, 0.4, 0.2), P(0.1 + i * 0.01, 0.45, 0.32)]), 66);
assert.ok(out.every((o) => o.state === "locked" && o.index === 0));

// 3. the signer moves around: still tracked
l = new SignerLock();
out = run(l, Array.from({ length: 40 }, (_, i) => [P(0.3 + i * 0.01, 0.4, 0.2)]));
assert.ok(out.every((o) => o.state === "locked" && o.index === 0));

// 4. two people cross paths with different sizes: no identity swap (the list order also flips)
l = new SignerLock();
const frames = Array.from({ length: 50 }, (_, i) => { const a = P(0.2 + i * 0.012, 0.4, 0.22), b = P(0.8 - i * 0.012, 0.4, 0.14); return i % 2 ? [a, b] : [b, a]; });
const ids = []; frames.forEach((f, i) => { const o = l.update(f, i * 66); ids.push(f[o.index].sw); });
assert.ok(ids.every((s) => s === 0.22));

// 5. signer leaves: lost, never jumps to the stranger, relocks on the same person coming back
l = new SignerLock();
l.update([P(0.3, 0.4, 0.2), P(0.8, 0.4, 0.12)], 0);
r = l.update([P(0.8, 0.4, 0.12)], 2000);                        // signer gone for 2 s
assert.equal(r.state, "lost"); assert.equal(r.index, -1);
r = l.update([P(0.8, 0.4, 0.12)], 2100);                        // the stranger is still not chosen
assert.equal(r.state, "lost");
r = l.update([P(0.33, 0.41, 0.2), P(0.8, 0.4, 0.12)], 2200);   // the signer returns
assert.equal(r.state, "locked"); assert.equal(r.index, 0);

// 6. tap selects the person, even the smaller one; a tap in empty space picks the nearest
l = new SignerLock({ mode: "tap" });
assert.equal(l.update([P(0.3, 0.4, 0.12), P(0.8, 0.4, 0.30)], 0).state, "searching");
l.select([0.3, 0.55]);
r = l.update([P(0.3, 0.4, 0.12), P(0.8, 0.4, 0.30)], 100);
assert.equal(r.index, 0); assert.equal(r.state, "locked");
l.select([0.6, 0.1]);
assert.equal(l.update([P(0.3, 0.4, 0.12), P(0.8, 0.4, 0.30)], 200).state, "locked");

// 7. off mode: no focus, first person, honest state
l = new SignerLock({ mode: "off" });
r = l.update([P(0.3, 0.4, 0.12), P(0.8, 0.4, 0.30)], 0);
assert.deepEqual([r.state, r.index], ["off", 0]);

// 8. tiny or null people are ignored
l = new SignerLock();
r = l.update([null, P(0.5, 0.4, 0.01), P(0.4, 0.4, 0.15)], 0);
assert.equal(r.index, 2);
assert.equal(personFromPose([[0.5, 0.2, 0], [0.4, 0.35, 0], [0.6, 0.35, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], [1, 0.2, 0.9, 1, 1, 1, 1]), null);

// 9. personFromPose corrects the aspect ratio
const pp = personFromPose([[0.5, 0.2, 0], [0.4, 0.35, 0], [0.6, 0.35, 0], [0, 0, 0], [0, 0, 0], [0.62, 0.7, 0], [0.38, 0.7, 0]], [1, 1, 1, 1, 1, 1, 1], 2);
assert.ok(Math.abs(pp.sw - 0.4) < 1e-9 && Math.abs(pp.anchor[0] - 1.0) < 1e-9);

// 10. hands: only the signer's hands are kept, assigned by the nearest wrist
const sig = P(0.3, 0.4, 0.2, [[0.45, 0.5], [0.15, 0.5]]);
const hs = [{ id: "far", wrist: [0.85, 0.5] }, { id: "right", wrist: [0.16, 0.52] }, { id: "left", wrist: [0.44, 0.48] }];
let f = filterHands(sig, hs);
assert.equal(f.L.id, "left"); assert.equal(f.R.id, "right");
assert.deepEqual(filterHands(sig, [{ id: "far", wrist: [0.85, 0.5] }]), { L: null, R: null });   // a bystander's hand
f = filterHands(sig, [{ id: "a", wrist: [0.15, 0.5] }, { id: "b", wrist: [0.16, 0.5] }]);         // two hands near one wrist: one is dropped
assert.equal(f.R.id, "a"); assert.equal(f.L, null);

// 11. wrists hidden: a hand inside the torso area is accepted, one outside is not
const hid = { ...sig, wrists: [null, null] };
assert.equal(filterHands(hid, [{ id: "n", wrist: [0.4, 0.6] }]).L.id, "n");
assert.deepEqual(filterHands(hid, [{ id: "x", wrist: [0.9, 0.6] }]), { L: null, R: null });
assert.deepEqual(filterHands(null, hs), { L: null, R: null });

// 12. focusedHand with MediaPipe-shaped data: two people, the signer is the closest one
const pose33 = (cx, cy, sw, raised) => {
  const lm = Array.from({ length: 33 }, () => ({ x: 0, y: 0, z: 0, visibility: 0 }));
  const set = (i, x, y) => { lm[i] = { x, y, z: 0, visibility: 0.99 }; };
  set(0, cx, cy - sw); set(11, cx + sw / 2, cy); set(12, cx - sw / 2, cy); set(13, cx + sw, cy + sw); set(14, cx - sw, cy + sw);
  set(15, cx + sw, cy + (raised ? 0.5 : 2) * sw); set(16, cx - sw, cy + (raised ? 0.5 : 2) * sw);
  return lm;
};
const hand = (x, y, score, label) => ({ landmarks: Array.from({ length: 21 }, (_, i) => ({ x: x + i * 0.001, y, z: 0 })), label, score });
const A = 1.0;   // aspect 1 keeps the numbers readable
const lockF = new SignerLock();
let fr = focusedHand(lockF, [pose33(0.2, 0.4, 0.14, true), pose33(0.75, 0.4, 0.26, true)],
  [hand(0.34, 0.47, 0.99, "Right"), hand(1.01, 0.53, 0.9, "Left")], A, 0);
assert.equal(fr.state, "locked"); assert.equal(fr.persons, 2);
assert.ok(Math.abs(fr.hand.landmarks[0].x - 1.01) < 1e-9, "the closest person's hand is used, not the bystander's better-scored one");
// the bystander signs close to the signer's wrist in the image: still the bystander's hand, still ignored
fr = focusedHand(lockF, [pose33(0.2, 0.4, 0.14, true), pose33(0.75, 0.4, 0.26, true)], [hand(0.34, 0.47, 0.99, "Right")], A, 66);
assert.equal(fr.hand, null);
assert.ok(fr.box[0] >= 0 && fr.box[2] <= 1 && fr.box[0] < fr.box[2]);
// no people detected -> searching, no hand
assert.equal(focusedHand(new SignerLock(), [], [hand(0.5, 0.5, 0.9, "Right")], A, 0).hand, null);
// focus off: the best hand overall, as before
const off = new SignerLock({ mode: "off" });
fr = focusedHand(off, [pose33(0.3, 0.4, 0.14), pose33(0.7, 0.4, 0.26)], [hand(0.3, 0.62, 0.99, "Right"), hand(0.96, 0.62, 0.9, "Left")], A, 0);
assert.equal(fr.state, "off"); assert.equal(fr.hand.score, 0.99);

console.log("signer lock: 12 test groups passed");
