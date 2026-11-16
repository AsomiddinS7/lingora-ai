#!/usr/bin/env python3
"""Train the Lingora static-sign classifier from collected landmark exports.

  python train/train.py --data "data/*.json" --signs signs.json --out models/run1

What it does
  1. Loads exports from collect/ (only samples with consent = true).
  2. Leave-one-signer-out (LOSO) evaluation: every signer is held out once.
  3. Reports accuracy with signer-level bootstrap intervals, per-sign recall,
     out-of-vocabulary ("none") rejection and a confidence-threshold table.
  4. Fits temperature scaling on the pooled held-out logits.
  5. Trains the final model on all signers and writes model_mlp.json for
     shared/mlp.js, plus report.json, CSV tables and a parity fixture.
"""
import argparse, csv, glob, json, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))
from features import features, augment_landmarks, FEATURE_DIM, FEATURE_VERSION  # noqa: E402

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import keras  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402


def load(paths):
    raw, hands, asp, y, who, dropped = [], [], [], [], [], 0
    for p in paths:
        d = json.load(open(p))
        if d.get("format") != "lingora-landmarks-v1":
            sys.exit(f"{p}: unknown format")
        for s in d["samples"]:
            if s.get("consent") is not True:
                dropped += 1
                continue
            raw.append(np.array(s["landmarks"], dtype=np.float64))
            hands.append(s.get("hand", "Right"))
            asp.append(s.get("aspect", 4 / 3))
            y.append(s["sign"])
            who.append(s["signer_id"])
    if dropped:
        print(f"dropped {dropped} samples without consent = true")
    return raw, hands, asp, np.array(y), np.array(who)


def feats(raw, hands, asp, idx, rng=None, k=0):
    X, Y = [], []
    for i in idx:
        X.append(features(raw[i], hands[i], asp[i])); Y.append(i)
        for _ in range(k):
            X.append(features(augment_landmarks(raw[i], asp[i], rng), hands[i], asp[i])); Y.append(i)
    return np.array(X, dtype=np.float32), np.array(Y)


def build(k):
    m = keras.Sequential([
        keras.layers.Input((FEATURE_DIM,)),
        keras.layers.Dense(128, activation="relu"), keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu"), keras.layers.Dropout(0.2),
        keras.layers.Dense(k)])
    m.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
    return m


def fit(X, y, k, epochs, seed):
    keras.utils.set_random_seed(seed)
    mean, std = X.mean(0), X.std(0) + 1e-6
    m = build(k)
    m.fit((X - mean) / std, y, epochs=epochs, batch_size=64, verbose=0)
    return m, mean, std


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def nll(T, logits, y):
    p = softmax(logits / T)
    return -np.log(p[np.arange(len(y)), y] + 1e-12).mean()


def ece(p, y, bins=15):
    conf, pred = p.max(1), p.argmax(1)
    edges, e = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def signer_bootstrap(correct, who, n=2000, seed=0):
    rng, ids = np.random.default_rng(seed), np.unique(who)
    per = {s: correct[who == s] for s in ids}
    vals = []
    for _ in range(n):
        pick = rng.choice(ids, len(ids), replace=True)
        vals.append(np.concatenate([per[s] for s in pick]).mean())
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="glob of export JSON files")
    ap.add_argument("--signs", default="signs.json")
    ap.add_argument("--out", default="models/run1")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--aug", type=int, default=4, help="augmented copies per sample")
    ap.add_argument("--threshold", type=float, default=0.80)
    a = ap.parse_args()

    paths = sorted(glob.glob(a.data))
    if not paths:
        sys.exit("no data files matched --data")
    raw, hands, asp, ylab, who = load(paths)
    cfg = json.load(open(a.signs))
    classes = [s["id"] for s in cfg["signs"]] + [cfg["none"]["id"]]
    unknown = sorted(set(ylab) - set(classes))
    if unknown:
        sys.exit(f"labels not in signs.json: {unknown}")
    cid = {c: i for i, c in enumerate(classes)}
    y = np.array([cid[v] for v in ylab]); k = len(classes)
    signers = sorted(set(who))
    print(f"{len(y)} samples, {len(signers)} signers, {k} classes")
    if len(signers) < 3:
        sys.exit("need at least 3 signers for leave-one-signer-out evaluation")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    # 1. Leave-one-signer-out
    logits_all, y_all, who_all = [], [], []
    for n, s in enumerate(signers):
        tr, te = np.where(who != s)[0], np.where(who == s)[0]
        Xtr, itr = feats(raw, hands, asp, tr, rng, a.aug)
        Xte, ite = feats(raw, hands, asp, te)
        m, mean, std = fit(Xtr, y[itr], k, a.epochs, n)
        lg = m.predict((Xte - mean) / std, verbose=0)
        acc = (lg.argmax(1) == y[te]).mean()
        print(f"  held out {s}: accuracy {acc:.3f} ({len(te)} samples)")
        logits_all.append(lg); y_all.append(y[te]); who_all.append(who[te])
    L, Y, W = np.vstack(logits_all), np.concatenate(y_all), np.concatenate(who_all)

    # 2. Temperature scaling on pooled held-out logits
    T = float(minimize_scalar(lambda t: nll(t, L, Y), bounds=(0.05, 10), method="bounded").x)
    P0, P = softmax(L), softmax(L / T)
    correct = (P.argmax(1) == Y)
    lo, hi = signer_bootstrap(correct, W)
    per_signer = {s: float(correct[W == s].mean()) for s in signers}

    # 3. Per-sign recall, confusion matrix, OOV rejection, threshold table
    none = cid[cfg["none"]["id"]]
    cm = np.zeros((k, k), int)
    for t, p in zip(Y, P.argmax(1)):
        cm[t, p] += 1
    recall = {c: float(cm[i, i] / cm[i].sum()) if cm[i].sum() else None for i, c in enumerate(classes)}
    rows = []
    for th in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        acc_mask = (P.max(1) >= th) & (P.argmax(1) != none)
        sign_mask = Y != none
        cov = float(acc_mask[sign_mask].mean())
        prec = float((P.argmax(1)[acc_mask & sign_mask] == Y[acc_mask & sign_mask]).mean()) if (acc_mask & sign_mask).any() else None
        oov_rej = float((~acc_mask[~sign_mask]).mean()) if (~sign_mask).any() else None
        rows.append({"threshold": th, "signs_accepted": cov, "accuracy_when_accepted": prec, "none_rejected": oov_rej})
    with open(out / "threshold_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(out / "confusion_matrix.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["true\\pred"] + classes)
        for i, c in enumerate(classes):
            w.writerow([c] + cm[i].tolist())

    report = {"feature_version": FEATURE_VERSION, "samples": int(len(y)), "signers": signers,
              "classes": classes, "temperature": T,
              "loso_accuracy_pooled": float(correct.mean()),
              "loso_accuracy_signer_bootstrap_95ci": [lo, hi],
              "per_signer_accuracy": per_signer, "per_sign_recall": recall,
              "ece_before": ece(P0, Y), "ece_after": ece(P, Y), "threshold_table": rows,
              "note": "Pooled accuracy counts every sample once; the interval resamples SIGNERS, "
                      "so it stays wide when few signers exist. That is intended."}
    json.dump(report, open(out / "report.json", "w"), indent=2)

    # 4. Final model on all signers
    Xall, iall = feats(raw, hands, asp, np.arange(len(y)), rng, a.aug)
    m, mean, std = fit(Xall, y[iall], k, a.epochs, 99)
    dense = [l for l in m.layers if isinstance(l, keras.layers.Dense)]
    layers = []
    for i, l in enumerate(dense):
        Wt, b = l.get_weights()
        layers.append({"W": Wt.tolist(), "b": b.tolist(), "act": "relu" if i < len(dense) - 1 else "linear"})
    model = {"format": "lingora-mlp-v1", "feature_version": FEATURE_VERSION, "classes": classes,
             "temperature": T, "default_threshold": a.threshold,
             "mean": mean.tolist(), "std": std.tolist(), "layers": layers}
    json.dump(model, open(out / "model_mlp.json", "w"))
    m.save(out / "model.keras")

    # 5. Parity fixture for the browser code
    pick = np.random.default_rng(1).choice(len(y), 8, replace=False)
    fx = []
    for i in pick:
        f = features(raw[i], hands[i], asp[i])
        z = m.predict(((f - mean) / std)[None], verbose=0)
        fx.append({"landmarks": raw[i].tolist(), "hand": hands[i], "aspect": asp[i],
                   "features": f.tolist(), "probs": softmax(z / T)[0].tolist()})
    json.dump(fx, open(out / "parity_fixture.json", "w"))

    print(f"\nLOSO accuracy {correct.mean():.3f}, signer-bootstrap 95% CI [{lo:.3f}, {hi:.3f}], T = {T:.2f}")
    print(f"ECE before {report['ece_before']:.3f} -> after {report['ece_after']:.3f}")
    print("per-sign recall:", {c: (round(v, 2) if v is not None else None) for c, v in recall.items()})
    print(f"wrote {out}/ (model_mlp.json, report.json, threshold_table.csv, confusion_matrix.csv)")


if __name__ == "__main__":
    main()
