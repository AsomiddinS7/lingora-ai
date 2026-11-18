// Focus mode: lock onto ONE signer and ignore everybody else in view.
// Pure logic (no browser APIs) so it is unit tested in Node.
//
// Per video frame, the recognizer gives update() every detected person (run the pose model with
// numPoses >= 3 and the hand model with numHands >= 4), and gets back which person is the signer.
// Then filterHands() keeps only the hands that belong to that person. Everything else is dropped
// at once: bystanders' landmarks are never buffered, stored or sent.
//
// All coordinates are aspect-corrected: x is multiplied by video width / height, so distances are
// the same in every direction. personFromPose() does that.

const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

/** pose: 7 points [nose, Lshoulder, Rshoulder, Lelbow, Relbow, Lwrist, Rwrist] as [x, y, z];
 *  vis: 7 visibilities. Returns a person summary, or null if the shoulders are not reliably visible. */
export function personFromPose(pose, vis, aspect = 4 / 3, minVis = 0.5) {
  if (!pose || !vis || vis[1] < minVis || vis[2] < minVis) return null;
  const p = (i) => [pose[i][0] * aspect, pose[i][1]];
  const ls = p(1), rs = p(2);
  return {
    anchor: [(ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2],
    sw: dist(ls, rs),
    nose: vis[0] >= 0.3 ? p(0) : null,
    wrists: [vis[5] >= 0.3 ? p(5) : null, vis[6] >= 0.3 ? p(6) : null],   // [subject's left, subject's right]
  };
}

export class SignerLock {
  /** mode: "auto" (the closest person) | "tap" (the person you choose with select()) | "off" (no focus) */
  constructor(o = {}) {
    this.mode = o.mode ?? "auto";
    this.minSw = o.minSw ?? 0.04;           // ignore people whose shoulders are tiny (far away)
    this.maxJump = o.maxJump ?? 1.5;        // shoulder widths the signer may move between frames
    this.speed = o.speed ?? 2.0;            // extra shoulder widths per second of allowed movement
    this.sizeWeight = o.sizeWeight ?? 2.0;  // penalty for a sudden change of apparent size
    this.lostMs = o.lostMs ?? 1500;         // no match for this long = the signer is lost
    this.ambiguousRatio = o.ambiguousRatio ?? 0.8;
    this.reset();
  }
  reset() { this.last = null; this.state = "searching"; this.tap = null; this.t = 0; }
  /** Choose the signer by a tap. point = [x, y] in the same aspect-corrected coordinates. */
  select(point) { this.tap = point; this.mode = "tap"; this.last = null; this.state = "searching"; }
  setMode(m) { this.mode = m; if (m !== "tap") this.tap = null; this.last = null; this.state = "searching"; }

  /** people: array of person summaries (or null for undetectable ones). Returns
   *  { state: "off"|"searching"|"locked"|"lost", index, persons, ambiguous } */
  update(people, tMs) {
    const list = people.map((p, i) => ({ p, i })).filter((x) => x.p && x.p.sw >= this.minSw);
    const persons = list.length;
    if (this.mode === "off") return { state: "off", index: list.length ? list[0].i : -1, persons, ambiguous: false };

    if (this.last && this.state === "locked") {
      const dt = Math.max(0, (tMs - this.last.t) / 1000);
      const allowed = this.maxJump + this.speed * Math.min(dt, 0.5);   // capped: a long gap must not excuse a jump to a stranger
      const scored = list.map(({ p, i }) => ({ p, i, cost: dist(p.anchor, this.last.anchor) / this.last.sw + this.sizeWeight * Math.abs(Math.log(p.sw / this.last.sw)) }))
        .sort((a, b) => a.cost - b.cost);
      if (scored.length && scored[0].cost <= allowed) {
        const b = scored[0];
        this.last = { anchor: b.p.anchor, sw: b.p.sw, t: tMs };
        return { state: "locked", index: b.i, persons, ambiguous: scored.length > 1 && scored[1].cost - b.cost < 0.15 };
      }
      if (tMs - this.last.t > this.lostMs) this.state = "lost";
      return { state: this.state, index: -1, persons, ambiguous: false };
    }

    if (this.state === "lost" && this.last) {      // only the same person coming back may relock; never a stranger
      const back = list.filter(({ p }) => dist(p.anchor, this.last.anchor) / this.last.sw <= 2.5 && Math.abs(Math.log(p.sw / this.last.sw)) <= 0.25)
        .sort((a, b) => dist(a.p.anchor, this.last.anchor) - dist(b.p.anchor, this.last.anchor));
      if (back.length) return this._lock(back[0], tMs, false, persons);
      return { state: "lost", index: -1, persons, ambiguous: false };
    }

    if (!list.length) return { state: "searching", index: -1, persons, ambiguous: false };
    if (this.tap) {                                 // the person whose body is nearest to the tap
      const box = (p) => (Math.abs(this.tap[0] - p.anchor[0]) <= 1.6 * p.sw && this.tap[1] >= p.anchor[1] - 1.6 * p.sw && this.tap[1] <= p.anchor[1] + 3.2 * p.sw);
      const hit = list.filter(({ p }) => box(p));
      const pool = hit.length ? hit : list;
      pool.sort((a, b) => dist(a.p.anchor, this.tap) / a.p.sw - dist(b.p.anchor, this.tap) / b.p.sw);
      this.tap = null;
      return this._lock(pool[0], tMs, false, persons);
    }
    if (this.mode === "tap") return { state: "searching", index: -1, persons, ambiguous: false };
    const bySize = [...list].sort((a, b) => b.p.sw - a.p.sw);   // auto: the closest person is the largest
    const amb = bySize.length > 1 && bySize[1].p.sw >= this.ambiguousRatio * bySize[0].p.sw;
    return this._lock(bySize[0], tMs, amb, persons);
  }
  _lock(c, tMs, ambiguous, persons) {
    this.last = { anchor: c.p.anchor, sw: c.p.sw, t: tMs }; this.state = "locked";
    return { state: "locked", index: c.i, persons, ambiguous };
  }
}

/** Keeps only the hands that belong to the signer. hands: [{ wrist: [x, y] (aspect-corrected), ... }].
 *  Returns { L, R } (a hand object or null). A hand farther than `reach` shoulder widths from the
 *  signer's wrists is somebody else's and is dropped. */
export function filterHands(person, hands, reach = 1.0) {
  const out = { L: null, R: null };
  if (!person) return out;
  const cand = [];
  hands.forEach((h) => {
    [0, 1].forEach((s) => {
      const w = person.wrists[s];
      if (w) { const d = dist(h.wrist, w) / person.sw; if (d <= reach) cand.push({ h, s, d }); }
    });
  });
  cand.sort((a, b) => a.d - b.d);
  const used = new Set();
  for (const c of cand) {
    const slot = c.s === 0 ? "L" : "R";
    if (out[slot] || used.has(c.h)) continue;
    out[slot] = c.h; used.add(c.h);
  }
  // wrists not visible (arms hidden): accept an unassigned hand inside the torso area, left or right by side
  for (const h of hands) {
    if (used.has(h)) continue;
    const [ax, ay] = person.anchor, sw = person.sw;
    const inside = Math.abs(h.wrist[0] - ax) <= 2.2 * sw && h.wrist[1] >= ay - 1.2 * sw && h.wrist[1] <= ay + 3.0 * sw;
    const hasWrists = person.wrists[0] || person.wrists[1];
    if (!inside || hasWrists) continue;
    const slot = h.wrist[0] >= ax ? "L" : "R";   // the subject's left appears on the image right (unmirrored)
    if (!out[slot]) { out[slot] = h; used.add(h); }
  }
  return out;
}

export const POSE7 = [0, 11, 12, 13, 14, 15, 16];   // nose, shoulders, elbows, wrists in a 33-point pose

/** One frame of focus. poses: MediaPipe pose results (arrays of 33 {x, y, z, visibility});
 *  hands: [{ landmarks: [21 {x, y, z}], label, score }]; aspect: video width / height.
 *  Returns { state, persons, ambiguous, hand, hands, box } where hand is the signer's best hand
 *  (or null), hands is { L, R }, and box is the signer's body area in normalized image coordinates. */
export function focusedHand(lock, poses, hands, aspect, tMs) {
  const people = poses.map((lm) => personFromPose(POSE7.map((i) => [lm[i].x, lm[i].y, lm[i].z]), POSE7.map((i) => lm[i].visibility ?? 1), aspect));
  const st = lock.update(people, tMs);
  const hs = hands.map((h) => ({ ...h, wrist: [h.landmarks[0].x * aspect, h.landmarks[0].y] }));
  const persons = people.filter(Boolean).length;
  const best = (f) => [f.L, f.R].filter(Boolean).sort((a, b) => b.score - a.score)[0] || null;
  if (st.state === "off") {                                  // focus is off: every hand counts, as before
    const sorted = [...hs].sort((a, b) => b.score - a.score);
    return { state: "off", persons, ambiguous: false, hand: sorted[0] || null, hands: { L: null, R: null }, box: null };
  }
  const person = st.index >= 0 ? people[st.index] : null;
  // each hand belongs to the person whose wrist is nearest (relative to that person's size), so a
  // bystander's hand that overlaps the signer in the image is not taken
  const owner = (h) => {
    let best = { c: Infinity, i: -1 };
    people.forEach((p, i) => { if (p) p.wrists.forEach((w) => { if (w) { const c = dist(h.wrist, w) / p.sw; if (c < best.c) best = { c, i }; } }); });
    return best;
  };
  const mine = hs.filter((h) => { const o = owner(h); return o.i === st.index || o.c > 1.0; });
  const f = filterHands(person, mine);
  const box = person ? [Math.max(0, (person.anchor[0] - 2.2 * person.sw) / aspect), Math.max(0, person.anchor[1] - 1.6 * person.sw),
                        Math.min(1, (person.anchor[0] + 2.2 * person.sw) / aspect), Math.min(1, person.anchor[1] + 3.2 * person.sw)] : null;
  return { state: st.state, persons, ambiguous: st.ambiguous, hand: best(f), hands: f, box };
}
