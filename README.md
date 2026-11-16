# Lingora AI: data collection and sign classifier (v0.1)

Static-sign recognition from hand landmarks. Everything runs on the phone; video is never stored or sent.

```
signs.json            sign vocabulary (PLACEHOLDER until the consultant confirms it)
collect/index.html    landmark collection page (MediaPipe Hand Landmarker, consent gate, JSON export)
train/train.py        leave-one-signer-out evaluation + final model + report tables
train/make_synthetic.py  fake data, ONLY to test the pipeline
shared/features.py|js    feature extraction v1 (identical in both, parity tested)
shared/mlp.js         dependency-free runtime for the trained model
tests/parity.mjs      checks the JS output equals the Python output
data/                 put exports here (git-ignored, keep private)
models/               put hand_landmarker.task here (optional, see below)
```

## 1. Collect data
1. Edit `signs.json` once the vocabulary is confirmed (ids lowercase_with_underscores).
2. Serve the folder over HTTPS or localhost (camera needs it): `python3 -m http.server 8000`, open `http://localhost:8000/collect/`. For phones, deploy the folder to GitHub Pages.
3. Optional: download `hand_landmarker.task` (MediaPipe, float16) into `models/` so the page does not depend on Google's server.
4. Per signer: tick consent (only with a signed form), enter a code (S01, never a name), choose the sign, press Record, hold the sign. Vary light, distance and angle between takes.
5. Record the `none` class too: relaxed hands, random shapes, transitions, signs outside the vocabulary.
6. Export JSON after every session into `data/`.

Targets: 100+ frames per sign, at least 3 signers (more is better), and at least one Deaf native signer kept as a held-out evaluation set.

## 2. Train and evaluate
```
pip install tensorflow-cpu numpy scipy
python train/train.py --data "data/*.json" --signs signs.json --out models/run1
node tests/parity.mjs models/run1/model_mlp.json models/run1/parity_fixture.json
```
Outputs in `models/run1/`: `model_mlp.json` (for `shared/mlp.js`), `report.json`, `threshold_table.csv`, `confusion_matrix.csv`.

Evaluation is leave-one-signer-out. Accuracy intervals resample signers, so they are wide with few signers; that is honest, not a bug. Samples from the same take are never split across train and test because whole signers are held out.

## 3. What was tested here
- Feature parity JS vs Python: exact on features, probabilities within 1e-7.
- Training, evaluation, temperature scaling and export on SYNTHETIC data (works; the accuracy numbers mean nothing).
- The collection page passed a syntax check only. It has NOT been run in a real browser yet. Test it on your phone first.

## 4. Known limits
- One hand, static signs only. Signs with movement need a sequence model (a different feature extractor, see the Slovo plan).
- Handedness labels come from MediaPipe on the unmirrored image; features canonicalize left and right, so results stay consistent as long as the camera setup is the same in collection and in the app.
- The model is a small MLP exported as JSON instead of TF.js: no converter needed, about 0.4 MB, 30 lines of JS. The website's SignClassifier interface is unaffected.
- Exports contain landmark numbers of real people. Keep them private; do not commit them to a public repository.

## 5. Use the model in the website (Level 1)
Copy `shared/` and the trained `model_mlp.json` into the site, then:
```js
import { MediaPipeSignClassifier } from "./shared/classifier.js";
const clf = new MediaPipeSignClassifier({ modelUrl: "./model/model_mlp.json" });
await clf.load();
clf.attach(videoElement, ({ label, confidence }) => {
  if (label === "none") showNotRecognized();   // a hand is visible but no clear sign
  else showSign(label, confidence);            // one event per confirmed sign
});
clf.detach();
```
- Pass the UNMIRRORED video element; mirror the preview with CSS only.
- Defaults: threshold from the model file (0.80), 8 agreeing frames, 1 dropped frame tolerated, 5 frames without a sign before the same sign can fire again, 20 unclear frames before "none".
- `tests/stable.test.mjs` covers the filter (run with `node tests/stable.test.mjs`). `classifier.js` itself has not been run in a browser yet.

## 6. Level 1 scope
One-hand static signs, confirmed by the signer. Not in scope: moving signs, whole phrases, face and head hints, pretraining.
