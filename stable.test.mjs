import assert from "node:assert/strict";
import { StableSigns } from "../shared/stable.js";

const run = (frames, opts = {}) => {
  const ev = [];
  const s = new StableSigns({ onEvent: (e) => ev.push(e), ...opts });
  frames.forEach((f, i) => s.push(f, i));
  return ev;
};
const F = (label, confidence = 0.95, n = 1) => Array(n).fill({ label, confidence });
const H = (n) => Array(n).fill(null);

// 1. A held sign fires once, not on every frame
let ev = run([...F("yes", 0.95, 40)]);
assert.equal(ev.length, 1); assert.equal(ev[0].label, "yes");

// 2. Needs minFrames agreement: 7 frames is not enough
assert.equal(run(F("yes", 0.95, 7)).length, 0);

// 3. One flickered frame inside the streak is tolerated
ev = run([...F("yes", 0.95, 5), ...F("none", 0.9, 1), ...F("yes", 0.95, 5)]);
assert.equal(ev.length, 1);

// 4. Two bad frames break the streak
ev = run([...F("yes", 0.95, 5), ...F("none", 0.9, 2), ...F("yes", 0.95, 5)]);
assert.equal(ev.filter((e) => e.type === "sign").length, 0);

// 5. Same sign fires again only after the hand is released
ev = run([...F("yes", 0.95, 12), ...H(6), ...F("yes", 0.95, 12)]);
assert.equal(ev.filter((e) => e.type === "sign").length, 2);
ev = run([...F("yes", 0.95, 12), ...H(2), ...F("yes", 0.95, 12)]);
assert.equal(ev.filter((e) => e.type === "sign").length, 1);

// 6. A different sign fires without releasing
ev = run([...F("yes", 0.95, 10), ...F("no", 0.95, 10)]);
assert.deepEqual(ev.map((e) => e.label), ["yes", "no"]);

// 7. Below the threshold never fires; a lingering unclear hand reports "uncertain" once
ev = run(F("yes", 0.6, 60));
assert.equal(ev.filter((e) => e.type === "sign").length, 0);
assert.equal(ev.filter((e) => e.type === "uncertain").length, 1);

// 8. Out-of-vocabulary pose (model says none) is reported as uncertain, never as a sign
ev = run(F("none", 0.99, 30));
assert.deepEqual(ev.map((e) => e.type), ["uncertain"]);

// 9. No hand: silence
assert.equal(run(H(100)).length, 0);

// 10. Uncertain re-arms after the hand leaves
ev = run([...F("none", 0.99, 25), ...H(3), ...F("none", 0.99, 25)]);
assert.equal(ev.filter((e) => e.type === "uncertain").length, 2);

// 11. Confidence reported is the mean over the streak
ev = run([...F("yes", 0.9, 4), ...F("yes", 1.0, 4)]);
assert.ok(Math.abs(ev[0].confidence - 0.95) < 1e-9);

console.log("11 stable-filter tests passed");
