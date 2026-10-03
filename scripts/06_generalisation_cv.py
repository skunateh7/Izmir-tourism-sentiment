"""
Step 6 — Generalisation experiment.

Compares two pooled cross-validation schemes over all 598 reviews, using each family's
configuration selected on the main validation set (no re-tuning):
  (a) Random stratified 5-fold CV  -> "unseen reviews from known attractions"
  (b) Leave-one-attraction-out CV  -> "reviews from an attraction never seen in training"
Inside every fold, 15% of the training portion (stratified) is held out for early stopping.
Neural models: 3 seeds per fold; out-of-fold probabilities are averaged across seeds.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.metrics import bootstrap_ci, full_metrics  # noqa: E402
from src.models import HP, glove_matrix, tfidf_model, train_neural, vader_labels, vader_scores  # noqa: E402
from src.preprocess import LABEL2ID, Vocab  # noqa: E402
from src.select import parse_neural, parse_tfidf, selected_configs  # noqa: E402

OUT = os.path.join(ROOT, "results/generalisation")
os.makedirs(OUT, exist_ok=True)
SEEDS = [42, 7, 13]
d = pd.read_csv(os.path.join(ROOT, "data/processed/reviews_publication_final.csv"))
d["Processed_Text"] = d.Processed_Text.fillna("")
Y = d.Final_Sentiment.map(LABEL2ID).values
SEL = {k: v for k, v in selected_configs().items() if k in ["TFIDF-LR", "TFIDF-SVM", "CNN", "BiLSTM"]}
print("Selected configs:", SEL, flush=True)
VADER_ALL = vader_labels(vader_scores(d.Original_Review_Text))

folds = {"random_5fold": [], "leave_one_attraction_out": []}
for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(d, Y):
    folds["random_5fold"].append((tr, te, "fold"))
for a in sorted(d.Attraction_Name.unique()):
    te = np.where(d.Attraction_Name == a)[0]
    tr = np.where(d.Attraction_Name != a)[0]
    folds["leave_one_attraction_out"].append((tr, te, a))

oof = {s: {f: np.full(len(d), -1) for f in ["VADER"] + list(SEL)} for s in folds}
t0 = time.time()
for scheme, fl in folds.items():
    for k, (tr, te, name) in enumerate(fl):
        tr_i, va_i = train_test_split(tr, test_size=0.15, stratify=Y[tr], random_state=42)
        oof[scheme]["VADER"][te] = VADER_ALL[te]
        for fam in ["TFIDF-LR", "TFIDF-SVM"]:
            C, ng, cw = parse_tfidf(SEL[fam])
            vec, clf = tfidf_model("lr" if fam == "TFIDF-LR" else "svm", C, ng, cw)
            clf.fit(vec.fit_transform(d.Processed_Text.iloc[tr]), Y[tr])  # deterministic: use full train fold
            oof[scheme][fam][te] = clf.predict(vec.transform(d.Processed_Text.iloc[te]))
        vocab = Vocab(HP["max_vocab"]).fit(d.Processed_Text.iloc[tr_i])
        enc = lambda idx: vocab.encode(d.Processed_Text.iloc[idx].tolist(), HP["max_len"])  # noqa: E731
        Xtr, Xva, Xte = enc(tr_i), enc(va_i), enc(te)
        G = None
        for fam, arch in [("CNN", "cnn"), ("BiLSTM", "bilstm")]:
            emb, cw = parse_neural(SEL[fam])
            if emb == "glove" and G is None:
                G, _ = glove_matrix(vocab)
            P = []
            for s in SEEDS:
                m, _ = train_neural(arch, Xtr, Y[tr_i], Xva, Y[va_i], len(vocab), s,
                                    G if emb == "glove" else None, cw)
                P.append(m.predict(Xte, verbose=0))
            oof[scheme][fam][te] = np.mean(P, 0).argmax(1)
        print(f"{scheme:26s} fold {k+1:2d}/{len(fl)} ({name}) n_test={len(te)}  {time.time()-t0:.0f}s", flush=True)

res = {"selected_configs": SEL, "seeds_per_fold": SEEDS, "schemes": {}}
rows = []
for scheme in folds:
    res["schemes"][scheme] = {}
    for fam, p in oof[scheme].items():
        m = full_metrics(Y, p)
        m["macro_f1_ci95"] = bootstrap_ci(Y, p)
        res["schemes"][scheme][fam] = m
        rows.append(dict(scheme=scheme, family=fam, macro_f1=m["macro_f1"], ci_lo=m["macro_f1_ci95"][0],
                         ci_hi=m["macro_f1_ci95"][1], accuracy=m["accuracy"],
                         f1_Negative=m["per_class"]["Negative"]["f1"], f1_Neutral=m["per_class"]["Neutral"]["f1"],
                         f1_Positive=m["per_class"]["Positive"]["f1"]))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "cv_comparison.csv"), index=False)

# per-attraction performance under LOAO (attractions with >= 20 reviews)
pa = []
for a, sub in d.groupby("Attraction_Name"):
    idx = sub.index.values
    for fam, p in oof["leave_one_attraction_out"].items():
        pr = oof["random_5fold"][fam][idx]
        pa.append(dict(attraction=a, category=sub.Attraction_Category.iloc[0], n=len(idx), family=fam,
                       loao_acc=float((p[idx] == Y[idx]).mean()), random_acc=float((pr == Y[idx]).mean()),
                       loao_macro_f1=full_metrics(Y[idx], p[idx])["macro_f1"] if len(idx) >= 20 else np.nan))
pd.DataFrame(pa).to_csv(os.path.join(OUT, "per_attraction_loao.csv"), index=False)
np.savez(os.path.join(OUT, "oof_predictions.npz"), y=Y, ids=d.Review_ID.values,
         **{f"{s}__{f}": p for s in oof for f, p in oof[s].items()})
json.dump(res, open(os.path.join(OUT, "generalisation_results.json"), "w"), indent=1)
print(pd.DataFrame(rows).round(3).to_string())
