// Decides when a stream of per-frame predictions becomes ONE sign event.
// Pure logic, no browser APIs, so it is unit tested in Node.
//
// push(frame): frame is null (no hand) or { label, confidence }.
// Events (via onEvent):
//   { type: "sign",      label, confidence, t }  a sign held steady long enough
//   { type: "uncertain", confidence, t }          a hand is present but no clear sign
export class StableSigns {
  constructor(o = {}) {
    this.threshold = o.threshold ?? 0.8;          // minimum confidence
    this.minFrames = o.minFrames ?? 8;            // consecutive frames that must agree
    this.tolerance = o.tolerance ?? 1;            // dropped frames allowed inside a streak
    this.releaseFrames = o.releaseFrames ?? 5;    // frames of no hand or none before the same sign may fire again
    this.uncertainFrames = o.uncertainFrames ?? 20; // frames of an unclear hand before "uncertain"
    this.onEvent = o.onEvent ?? (() => {});
    this.reset();
  }
  reset() {
    this.cand = null; this.count = 0; this.sum = 0; this.miss = 0;
    this.fired = null; this.idle = 0; this.unsure = 0; this.unsureSent = false;
  }
  push(frame, t = Date.now()) {
    const hand = !!frame;
    const good = hand && frame.label !== "none" && frame.confidence >= this.threshold;
    if (good) {
      this.idle = 0; this.unsure = 0; this.unsureSent = false;
      if (frame.label !== this.cand) { this.cand = frame.label; this.count = 0; this.sum = 0; }
      this.miss = 0; this.count++; this.sum += frame.confidence;
      if (this.count >= this.minFrames && this.fired !== this.cand) {
        this.fired = this.cand;
        this.onEvent({ type: "sign", label: this.cand, confidence: this.sum / this.count, t });
      }
      return;
    }
    if (this.cand !== null && this.miss < this.tolerance) this.miss++;      // brief dropout
    else { this.cand = null; this.count = 0; this.sum = 0; this.miss = 0; }
    if (hand && frame.label !== "none") this.idle = 0;                      // low confidence, still holding
    else if (++this.idle >= this.releaseFrames) this.fired = null;          // hand gone or none: re-arm
    if (hand) {
      if (++this.unsure >= this.uncertainFrames && !this.unsureSent) {
        this.unsureSent = true;
        this.onEvent({ type: "uncertain", confidence: frame.confidence, t });
      }
    } else { this.unsure = 0; this.unsureSent = false; }
  }
}
