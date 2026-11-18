// SignClassifier for the website: camera video in, one event per confirmed sign out.
//
//   const clf = new MediaPipeSignClassifier({
//     modelUrl: "./model/model_mlp.json",
//     onFocus: (f) => showFocus(f),            // optional, see "Focus mode"
//   });
//   await clf.load();
//   clf.attach(videoElement, ({ label, confidence }) => { ... });
//   clf.detach();
//
// onSign receives { label, confidence } for a confirmed sign, or
// { label: "none", confidence } when a hand is visible but no sign is clear
// (show "Not recognized. Try again or choose a phrase").
//
// IMPORTANT: feed the UNMIRRORED video element. Mirror the preview with CSS only.
//
// Focus mode (two or more people in view)
//   Default "auto": locks onto the closest person and ignores everybody else. The pose model finds
//   all people, SignerLock follows the signer from frame to frame, and only the signer's hands are used.
//   clf.setFocus({ mode: "auto" })                       the closest person
//   clf.setFocus({ mode: "tap", point: [x, y] })         the person at a point; x, y are 0..1 in the
//                                                        UNMIRRORED video (if the preview is mirrored: x = 1 - tapX / width)
//   clf.setFocus({ mode: "off" })                        no focus (old behavior: best hand in view)
//   onFocus({ state: "searching" | "locked" | "lost" | "off", persons, ambiguous, box })
//     box = [x0, y0, x1, y1] of the locked person in unmirrored 0..1 coordinates, for a highlight.
//     state "lost": ask the signer to tap themselves. ambiguous: two people look equally close, suggest a tap.
//   If the pose model cannot load, focus is switched off and onFocus reports state "off".
import { HandLandmarker, PoseLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/vision_bundle.mjs";
import { features } from "./features.js";
import { MLP } from "./mlp.js";
import { StableSigns } from "./stable.js";
import { SignerLock, focusedHand } from "./signer_lock.js";

const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm";
const HAND_MODEL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const POSE_MODEL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

export class MediaPipeSignClassifier {
  constructor(opts = {}) {
    this.o = { modelUrl: "./model/model_mlp.json", handModelUrl: HAND_MODEL, poseModelUrl: POSE_MODEL, wasmUrl: WASM, focus: "auto", ...opts };
    this.running = false;
    this.lock = new SignerLock({ mode: this.o.focus });
    this.lastFocus = null;
  }
  async load() {
    this.mlp = await MLP.load(this.o.modelUrl);
    const fileset = await FilesetResolver.forVisionTasks(this.o.wasmUrl);
    const mk = async (cls, modelAssetPath, extra) => {
      const base = { runningMode: "VIDEO", ...extra };
      try { return await cls.createFromOptions(fileset, { baseOptions: { modelAssetPath, delegate: "GPU" }, ...base }); }
      catch { return await cls.createFromOptions(fileset, { baseOptions: { modelAssetPath, delegate: "CPU" }, ...base }); }
    };
    this.lm = await mk(HandLandmarker, this.o.handModelUrl, { numHands: 4 });          // hands of several people
    if (this.o.focus !== "off") {
      try { this.pose = await mk(PoseLandmarker, this.o.poseModelUrl, { numPoses: 3 }); }
      catch { this.pose = null; this.lock.setMode("off"); this._report({ state: "off", persons: 0, ambiguous: false, box: null }); }
    }
  }
  /** Change who is followed. See the header comment. */
  setFocus({ mode = "auto", point = null } = {}) {
    if (!this.pose && mode !== "off") return;                       // no pose model: nothing to focus with
    if (mode === "tap" && point) this.lock.select([point[0] * this._aspect(), point[1]]);
    else this.lock.setMode(mode);
    if (this.stable) this.stable.reset();
  }
  attach(video, onSign) {
    if (!this.lm) throw new Error("Call load() first");
    this.video = video;
    this.stable = new StableSigns({
      threshold: this.o.threshold ?? this.mlp.m.default_threshold ?? 0.8,
      minFrames: this.o.minFrames, tolerance: this.o.tolerance,
      releaseFrames: this.o.releaseFrames, uncertainFrames: this.o.uncertainFrames,
      onEvent: (e) => onSign(e.type === "sign"
        ? { label: e.label, confidence: e.confidence }
        : { label: "none", confidence: e.confidence }),
    });
    this.running = true; this.last = -1; this.errors = 0;
    this._next();
  }
  detach() {
    this.running = false;
    if (this.stable) this.stable.reset();
  }
  _aspect() { return this.video && this.video.videoHeight ? this.video.videoWidth / this.video.videoHeight : 4 / 3; }
  _report(f) {
    const key = `${f.state}|${f.persons}|${f.ambiguous}`;
    if (key !== this.lastFocus || f.state === "locked") {          // changes at once, the box while locked
      this.lastFocus = key;
      if (this.o.onFocus) this.o.onFocus(f);
    }
  }
  _next() {
    if (!this.running) return;
    const v = this.video;
    if (v.requestVideoFrameCallback) v.requestVideoFrameCallback(() => this._tick());
    else requestAnimationFrame(() => this._tick());
  }
  _tick() {
    if (!this.running) return;
    try {
      const v = this.video;
      if (v.readyState >= 2 && v.currentTime !== this.last) {
        this.last = v.currentTime;
        const now = performance.now(), aspect = this._aspect();
        const res = this.lm.detectForVideo(v, now);
        let hand = null, hands = (res.landmarks || []).map((landmarks, i) => ({
          landmarks, label: res.handedness[i][0].categoryName, score: res.handedness[i][0].score }));
        if (this.pose) {
          const f = focusedHand(this.lock, (this.pose.detectForVideo(v, now).landmarks) || [], hands, aspect, now);
          this._report({ state: f.state, persons: f.persons, ambiguous: f.ambiguous, box: f.box });
          hand = f.hand;
        } else {
          hand = [...hands].sort((a, b) => b.score - a.score)[0] || null;      // old behavior
        }
        if (!hand) this.stable.push(null);
        else {
          const lm = hand.landmarks.map((p) => [p.x, p.y, p.z]);
          const r = this.mlp.predict(features(lm, hand.label, aspect));
          this.stable.push({ label: r.label, confidence: r.confidence });
          if (this.o.onFrame) this.o.onFrame({ landmarks: hand.landmarks, hand: hand.label, label: r.label, confidence: r.confidence });
        }
      }
      this.errors = 0;
    } catch (e) {
      if (++this.errors > 30) { this.running = false; if (this.o.onError) this.o.onError(e); return; }
    }
    this._next();
  }
}
