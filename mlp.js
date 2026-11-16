// Tiny MLP runtime (no dependencies). Loads the JSON written by train.py.
// logits -> divide by temperature -> softmax.
export class MLP {
  constructor(model) {
    if (model.format !== "lingora-mlp-v1") throw new Error("Unknown model format");
    this.m = model;
    this.classes = model.classes;
  }
  static async load(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error("Model file not found: " + url);
    return new MLP(await r.json());
  }
  /** x: Float32Array of features. Returns { label, confidence, probs } */
  predict(x) {
    const { mean, std, layers, temperature } = this.m;
    let a = Array.from(x, (v, i) => (v - mean[i]) / (std[i] || 1));
    for (const L of layers) {
      const out = new Array(L.b.length);
      for (let j = 0; j < out.length; j++) {
        let s = L.b[j];
        for (let i = 0; i < a.length; i++) s += a[i] * L.W[i][j];
        out[j] = L.act === "relu" ? Math.max(0, s) : s;
      }
      a = out;
    }
    const z = a.map((v) => v / temperature);
    const mx = Math.max(...z);
    const e = z.map((v) => Math.exp(v - mx));
    const sum = e.reduce((p, c) => p + c, 0);
    const probs = e.map((v) => v / sum);
    let best = 0;
    for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
    return { label: this.classes[best], confidence: probs[best], probs };
  }
}
