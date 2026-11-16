// SignClassifier for the website: camera video in, one event per confirmed sign out.
//
//   const clf = new MediaPipeSignClassifier({ modelUrl: "./model/model_mlp.json" });
//   await clf.load();
//   clf.attach(videoElement, ({ label, confidence }) => { ... });
//   clf.detach();
//
// onSign receives { label, confidence } for a confirmed sign, or
// { label: "none", confidence } when a hand is visible but no sign is clear
// (show "Not recognized. Try again or choose a phrase").
//
// IMPORTANT: feed the UNMIRRORED video element. Mirror the preview with CSS only.
import { HandLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/vision_bundle.mjs";
import { features } from "./features.js";
import { MLP } from "./mlp.js";
import { StableSigns } from "./stable.js";

const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm";
const HAND_MODEL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

export class MediaPipeSignClassifier {
  constructor(opts = {}) {
    this.o = { modelUrl: "./model/model_mlp.json", handModelUrl: HAND_MODEL, wasmUrl: WASM, ...opts };
    this.running = false;
  }
  async load() {
    this.mlp = await MLP.load(this.o.modelUrl);
    const fileset = await FilesetResolver.forVisionTasks(this.o.wasmUrl);
    const mk = (delegate) => HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: this.o.handModelUrl, delegate }, runningMode: "VIDEO", numHands: 2 });
    try { this.lm = await mk("GPU"); } catch { this.lm = await mk("CPU"); }
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
        const res = this.lm.detectForVideo(v, performance.now());
        let best = -1, score = -1;
        (res.landmarks || []).forEach((_, i) => { const s = res.handedness?.[i]?.[0]?.score ?? 0; if (s > score) { score = s; best = i; } });
        if (best < 0) this.stable.push(null);
        else {
          const lm = res.landmarks[best].map((p) => [p.x, p.y, p.z]);
          const hand = res.handedness[best][0].categoryName;
          const r = this.mlp.predict(features(lm, hand, v.videoWidth / v.videoHeight));
          this.stable.push({ label: r.label, confidence: r.confidence });
          if (this.o.onFrame) this.o.onFrame({ landmarks: res.landmarks[best], hand, label: r.label, confidence: r.confidence });
        }
      }
      this.errors = 0;
    } catch (e) {
      if (++this.errors > 30) { this.running = false; if (this.o.onError) this.o.onError(e); return; }
    }
    this._next();
  }
}
