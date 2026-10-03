"""
Step 4 — Main experiment on the locked stratified split (418 / 90 / 90).

Families and the configuration space searched ON VALIDATION ONLY:
  VADER          : standard thresholds (±0.05) and neutral band tuned on validation
  TF-IDF + LR    : C × n-gram range × class weighting
  TF-IDF + SVM   : C × n-gram range × class weighting
  CNN, BiLSTM    : {random, GloVe-100d} embeddings × {unweighted, class-weighted} × 5 seeds
The test set is scored for every configuration (for transparency) but never used to select one.
"""
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.metrics import full_metrics  # noqa: E402
from src.models import (HP, glove_matrix, train_neural, tfidf_model, tune_vader,  # noqa: E402
                        vader_labels, vader_scores)
from src.preprocess import LABEL2ID, Vocab  # noqa: E402

OUT = os.path.join(ROOT, "results/main")
os.makedirs(os.path.join(OUT, "models"), exist_ok=True)
SEEDS = [42, 7, 13, 99, 2024]

d = pd.read_csv(os.path.join(ROOT, "data/processed/reviews_publication_final.csv"))
d["Processed_Text"] = d.Processed_Text.fillna("")
sp = json.load(open(os.path.join(ROOT, "data/processed/split_main.json")))
part = {k: d[d.Review_ID.isin(sp[f"{k}_ids"])].sort_values("Review_ID").reset_index(drop=True)
        for k in ["train", "val", "test"]}
y = {k: part[k].Final_Sentiment.map(LABEL2ID).values for k in part}
runs, preds = [], {"test_ids": part["test"].Review_ID.tolist(), "y_test": y["test"].tolist()}


def log(family, config, seed, pv, pt, extra=None):
    mv, mt = full_metrics(y["val"], pv), full_metrics(y["test"], pt)
    r = dict(family=family, config=config, seed=seed, val_macro_f1=mv["macro_f1"], val_acc=mv["accuracy"],
             test_macro_f1=mt["macro_f1"], test_acc=mt["accuracy"], test_weighted_f1=mt["weighted_f1"],
             test_macro_p=mt["macro_precision"], test_macro_r=mt["macro_recall"],
             **{f"test_f1_{c}": mt["per_class"][c]["f1"] for c in mt["per_class"]},
             **{f"test_recall_{c}": mt["per_class"][c]["recall"] for c in mt["per_class"]})
    if extra:
        r.update(extra)
    runs.append(r)
    print(f"{family:8s} {config:34s} seed={seed!s:5s} val={r['val_macro_f1']:.3f} test={r['test_macro_f1']:.3f}", flush=True)


t0 = time.time()
# ---------------- VADER (raw original text: VADER is designed for unprocessed text)
cv, ct = vader_scores(part["val"].Original_Review_Text), vader_scores(part["test"].Original_Review_Text)
log("VADER", "standard(-0.05,0.05)", "-", vader_labels(cv), vader_labels(ct))
preds["VADER|standard(-0.05,0.05)"] = {"pred": vader_labels(ct).tolist()}
_, lo, hi = tune_vader(cv, y["val"])
log("VADER", f"val-tuned({lo},{hi})", "-", vader_labels(cv, lo, hi), vader_labels(ct, lo, hi))
preds[f"VADER|val-tuned({lo},{hi})"] = {"pred": vader_labels(ct, lo, hi).tolist()}

# ---------------- TF-IDF classical baselines
for kind in ["lr", "svm"]:
    for C, ng, cw in itertools.product([0.1, 0.3, 1, 3, 10, 30], [(1, 1), (1, 2)], [None, "balanced"]):
        vec, clf = tfidf_model(kind, C, ng, cw)
        Xtr = vec.fit_transform(part["train"].Processed_Text)
        clf.fit(Xtr, y["train"])
        pv = clf.predict(vec.transform(part["val"].Processed_Text))
        pt = clf.predict(vec.transform(part["test"].Processed_Text))
        cfg = f"C={C},ngram={ng[0]}-{ng[1]},cw={'bal' if cw else 'none'}"
        fam = "TFIDF-LR" if kind == "lr" else "TFIDF-SVM"
        log(fam, cfg, "-", pv, pt)
        preds[f"{fam}|{cfg}"] = {"pred": pt.tolist()}

# ---------------- neural models
vocab = Vocab(HP["max_vocab"]).fit(part["train"].Processed_Text)
json.dump(vocab.itos, open(os.path.join(OUT, "models/vocab.json"), "w"))
X = {k: vocab.encode(part[k].Processed_Text.tolist(), HP["max_len"]) for k in part}
G, cov = glove_matrix(vocab)
meta = {"vocab_size": len(vocab), "glove_coverage_of_train_vocab": cov,
        "oov_rate_test_tokens": float((X["test"] == 1).sum() / max(1, (X["test"] > 0).sum()))}
curves = {}
for arch in ["cnn", "bilstm"]:
    for emb, cw in itertools.product(["random", "glove"], [False, True]):
        cfg = f"emb={emb},cw={'bal' if cw else 'none'}"
        fam = "CNN" if arch == "cnn" else "BiLSTM"
        probs = []
        for s in SEEDS:
            m, cb = train_neural(arch, X["train"], y["train"], X["val"], y["val"], len(vocab), s,
                                 G if emb == "glove" else None, cw)
            pv, ptp = m.predict(X["val"], verbose=0).argmax(1), m.predict(X["test"], verbose=0)
            probs.append(ptp)
            log(fam, cfg, s, pv, ptp.argmax(1), {"best_epoch": cb.best_epoch, "epochs_run": len(cb.hist)})
            curves[f"{fam}|{cfg}|{s}"] = cb.hist
            preds[f"{fam}|{cfg}|seed{s}"] = {"pred": ptp.argmax(1).tolist(), "prob": np.round(ptp, 5).tolist()}
            m.save(os.path.join(OUT, f"models/{fam}_{emb}_{'bal' if cw else 'none'}_seed{s}.keras"))
        ens = np.mean(probs, 0)
        preds[f"{fam}|{cfg}|ensemble"] = {"pred": ens.argmax(1).tolist(), "prob": np.round(ens, 5).tolist()}
        print(f"   elapsed {time.time()-t0:.0f}s", flush=True)

R = pd.DataFrame(runs)
R.to_csv(os.path.join(OUT, "all_runs.csv"), index=False)
json.dump(preds, open(os.path.join(OUT, "test_predictions.json"), "w"))
json.dump(curves, open(os.path.join(OUT, "training_curves.json"), "w"))
meta["hyperparameters"] = {"common": HP}
from src.models import CNN_HP, LSTM_HP  # noqa: E402
meta["hyperparameters"]["cnn"] = {k: list(v) if isinstance(v, tuple) else v for k, v in CNN_HP.items()}
meta["hyperparameters"]["bilstm"] = {k: list(v) if isinstance(v, tuple) else v for k, v in LSTM_HP.items()}
meta["seeds"] = SEEDS
meta["runtime_seconds"] = round(time.time() - t0)
json.dump(meta, open(os.path.join(OUT, "experiment_meta.json"), "w"), indent=1)
print("DONE", meta)
