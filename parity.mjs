// node tests/parity.mjs models/run1/model_mlp.json models/run1/parity_fixture.json
import { readFileSync } from "node:fs";
import { features } from "../shared/features.js";
import { MLP } from "../shared/mlp.js";

const [modelPath, fixturePath] = process.argv.slice(2);
const mlp = new MLP(JSON.parse(readFileSync(modelPath, "utf8")));
const fx = JSON.parse(readFileSync(fixturePath, "utf8"));
let worstF = 0, worstP = 0;
for (const t of fx) {
  const f = features(t.landmarks, t.hand, t.aspect);
  t.features.forEach((v, i) => { worstF = Math.max(worstF, Math.abs(v - f[i])); });
  const r = mlp.predict(f);
  r.probs.forEach((v, i) => { worstP = Math.max(worstP, Math.abs(v - t.probs[i])); });
}
console.log(`max feature diff ${worstF.toExponential(2)}, max probability diff ${worstP.toExponential(2)}`);
if (worstF > 1e-4 || worstP > 1e-3) { console.error("PARITY FAILED"); process.exit(1); }
console.log("PARITY OK");
