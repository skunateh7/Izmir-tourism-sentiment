"""
Step 14 - Nested cross-validation for the TF-IDF models (response to review; CPU, ~minutes).

Same outer folds as Step 6 (random stratified 5-fold, seed 42; leave-one-entity-out over 14 entities).
Within each outer fold the TF-IDF configuration (C x n-gram range x class weighting, the Step-4 grid) is chosen
using ONLY the outer training data, by pooled inner macro-F1:
  random scheme -> inner stratified 5-fold CV;   LOEO scheme -> inner leave-one-entity-out over the 13 training entities.
The held-out reviews/entity therefore never influence configuration choice.
Output: results/generalisation/nested_tfidf.json and nested_tfidf_selected_configs.csv
"""
import itertools, json, os, sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.preprocess import LABEL2ID  # noqa: E402

GRID = list(itertools.product([0.1, 0.3, 1, 3, 10, 30], [(1, 1), (1, 2)], [None, "balanced"]))
mf1 = lambda y, p: f1_score(y, p, average="macro")  # noqa: E731


def model(kind, C, ng, cw):
    vec = TfidfVectorizer(ngram_range=ng, min_df=1, sublinear_tf=True)
    clf = LogisticRegression(C=C, max_iter=5000, class_weight=cw) if kind == "lr" else LinearSVC(C=C, class_weight=cw)
    return vec, clf


def fitpred(kind, cfg, Xtr, ytr, Xte):
    vec, clf = model(kind, *cfg)
    clf.fit(vec.fit_transform(Xtr), ytr)
    return clf.predict(vec.transform(Xte))


d = pd.read_csv(os.path.join(ROOT, "data/processed/reviews_publication_final.csv"))
X = d.Processed_Text.fillna("").values
Y = d.Final_Sentiment.map(LABEL2ID).values
ENT = d.Attraction_Name.values

outer = {"random_5fold": [(tr, te, "fold") for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(X, Y)],
         "leave_one_attraction_out": [(np.where(ENT != a)[0], np.where(ENT == a)[0], a) for a in sorted(set(ENT))]}


def inner_splits(scheme, tr):
    if scheme == "random_5fold":
        return [(tr[a], tr[b]) for a, b in StratifiedKFold(5, shuffle=True, random_state=7).split(tr, Y[tr])]
    return [(tr[ENT[tr] != e], tr[ENT[tr] == e]) for e in sorted(set(ENT[tr]))]


def boot_ci(y, p, B=2000, seed=1):
    r = np.random.default_rng(seed); n = len(y)
    s = [mf1(y[i], p[i]) for i in (r.integers(0, n, n) for _ in range(B))]
    return [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]


res, rows = {}, []
for kind, fam in [("lr", "TFIDF-LR"), ("svm", "TFIDF-SVM")]:
    for scheme, folds in outer.items():
        oof = np.full(len(Y), -1)
        for tr, te, name in folds:
            inner = inner_splits(scheme, tr)
            idx = np.concatenate([b for _, b in inner])

            def score(cfg):
                pred = np.full(len(Y), -1)
                for itr, iva in inner:
                    pred[iva] = fitpred(kind, cfg, X[itr], Y[itr], X[iva])
                return mf1(Y[idx], pred[idx])
            sc = Parallel(n_jobs=-1)(delayed(score)(c) for c in GRID)
            j = int(np.argmax(sc)); best, bestcfg = sc[j], GRID[j]
            oof[te] = fitpred(kind, bestcfg, X[tr], Y[tr], X[te])
            rows.append(dict(family=fam, scheme=scheme, fold=name, C=bestcfg[0], ngram=f"{bestcfg[1][0]}-{bestcfg[1][1]}",
                             cw="bal" if bestcfg[2] else "none", inner_macro_f1=round(best, 4)))
        res.setdefault(fam, {})[scheme] = {"macro_f1": float(mf1(Y, oof)), "ci": boot_ci(Y, oof), "oof": oof.tolist()}
        print(fam, scheme, round(mf1(Y, oof), 3), flush=True)
    a, b = np.array(res[fam]["random_5fold"]["oof"]), np.array(res[fam]["leave_one_attraction_out"]["oof"])
    r = np.random.default_rng(2); n = len(Y)
    diffs = [mf1(Y[i], a[i]) - mf1(Y[i], b[i]) for i in (r.integers(0, n, n) for _ in range(2000))]
    res[fam]["drop"] = {"value": float(mf1(Y, a) - mf1(Y, b)), "ci": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}
OOF = {}
G = pd.read_csv(os.path.join(ROOT, "results/generalisation/cv_comparison.csv")).set_index(["family", "scheme"])
for fam in res:
    for sch in outer:
        res[fam][sch]["non_nested_macro_f1"] = float(G.loc[(fam, sch), "macro_f1"])
        OOF[f"{fam}|{sch}"] = np.array(res[fam][sch].pop("oof"))
out = os.path.join(ROOT, "results/generalisation")
# DistilBERT (non-nested; its two best configurations differed by 0.001 on validation) vs nested TF-IDF
P = np.load(os.path.join(out, "oof_predictions.npz"), allow_pickle=True)
key = [k for k in P.files if "DistilBERT" in k]
for sch in outer:
    kk = [k for k in key if sch in k]
    if not kk:
        continue
    db = P[kk[0]]
    for fam in res:
        nb = OOF[f"{fam}|{sch}"]; r = np.random.default_rng(3); n = len(Y)
        ds = [mf1(Y[i], db[i]) - mf1(Y[i], nb[i]) for i in (r.integers(0, n, n) for _ in range(2000))]
        res[fam][sch]["distilbert_minus_nested"] = {"value": float(mf1(Y, db) - mf1(Y, nb)),
                                                    "ci": [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))]}
np.savez(os.path.join(out, "nested_tfidf_oof.npz"), **{k.replace("|", "__"): v for k, v in OOF.items()})
json.dump(res, open(os.path.join(out, "nested_tfidf.json"), "w"), indent=1)
pd.DataFrame(rows).to_csv(os.path.join(out, "nested_tfidf_selected_configs.csv"), index=False)
print(json.dumps(res, indent=1))
