"""
Step 16 — Review round 4: equivalence of random-CV and LOEO performance; entity-level (cluster-aware) inference.

(a) For each model and protocol, the random-CV minus LOEO macro-F1 difference on the same 598 reviews, with
    paired bootstrap intervals (reviews resampled) and cluster bootstrap intervals (the 14 entities resampled with
    replacement, all their reviews kept together). Equivalence is assessed by two one-sided tests (TOST) against a
    margin of 0.05 macro-F1: equivalent at alpha = 0.05 if the 90% interval of the drop lies within (-0.05, 0.05).
    The smallest margin supported is the larger absolute end of the 90% interval.
(b) Category differences in observed ratings with entities as the unit: permutation test that reassigns category
    labels among the 14 entities (category sizes fixed), statistic = between-category variance of entity-level
    negative share and of net sentiment (unweighted and rating-count weighted). 20,000 permutations.
Outputs: results/review2/equivalence.json, equivalence_table.csv, category_permutation.json
"""
import json
import os
import unicodedata

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results/review2")
G = os.path.join(ROOT, "results/generalisation")
d = pd.read_csv(os.path.join(ROOT, "data/processed/reviews_publication_final.csv"))
Z = np.load(os.path.join(G, "oof_predictions.npz"))
Y = Z["y"]
assert (Z["ids"] == d.Review_ID.values).all()
ENT = d.Attraction_Name.values
mf1 = lambda y, p: f1_score(y, p, average="macro", labels=[0, 1, 2], zero_division=0)
B, MARGIN = 4000, 0.05

# ---- assemble OOF predictions per protocol
P = {}
for f in ["VADER", "TFIDF-LR", "TFIDF-SVM", "CNN", "BiLSTM", "DistilBERT"]:
    P[(f, "fixed")] = (Z[f"random_5fold__{f}"], Z[f"leave_one_attraction_out__{f}"])
N = np.load(os.path.join(G, "nested_tfidf_oof.npz"))
for f in ["TFIDF-LR", "TFIDF-SVM"]:
    P[(f, "nested")] = (N[f"{f}__random_5fold"], N[f"{f}__leave_one_attraction_out"])
# nested DistilBERT, rebuilt from the per-fold files of step 5c
CD = os.path.join(ROOT, "results/transformer_config_cv")
CFG = ["lr=2e-05,cw=none", "lr=2e-05,cw=bal", "lr=5e-05,cw=none", "lr=5e-05,cw=bal"]
from sklearn.model_selection import StratifiedKFold  # noqa: E402
nest = {}
for sch, nf in [("random_5fold", 5), ("leave_one_attraction_out", 14)]:
    o = np.full(len(d), -1)
    for k in range(nf):
        zs = {c: np.load(os.path.join(CD, f"{c}__{sch}__{k:02d}.npz")) for c in CFG}
        best = max(CFG, key=lambda c: zs[c]["val_f1"].mean())
        o[zs[best]["te"]] = zs[best]["prob"].argmax(1)
    assert (o >= 0).all()
    nest[sch] = o
P[("DistilBERT", "nested")] = (nest["random_5fold"], nest["leave_one_attraction_out"])

rng = np.random.default_rng(0)
idx_rev = [rng.integers(0, len(Y), len(Y)) for _ in range(B)]
ents = np.unique(ENT)
by_ent = {e: np.where(ENT == e)[0] for e in ents}
idx_ent = [np.concatenate([by_ent[e] for e in rng.choice(ents, len(ents), replace=True)]) for _ in range(B)]


def drops(a, b, idxs):
    return np.array([mf1(Y[i], a[i]) - mf1(Y[i], b[i]) for i in idxs])


rows = []
for (f, prot), (a, b) in P.items():
    if f == "VADER":
        continue
    dr = mf1(Y, a) - mf1(Y, b)
    r_rev, r_ent = drops(a, b, idx_rev), drops(a, b, idx_ent)
    row = {"model": f, "protocol": prot, "random_cv": mf1(Y, a), "loeo": mf1(Y, b), "drop": dr}
    for lab, r in [("review", r_rev), ("entity", r_ent)]:
        lo95, hi95 = np.percentile(r, [2.5, 97.5]); lo90, hi90 = np.percentile(r, [5, 95])
        row.update({f"ci95_{lab}": [float(lo95), float(hi95)], f"ci90_{lab}": [float(lo90), float(hi90)],
                    f"tost_equivalent_{lab}": bool(lo90 > -MARGIN and hi90 < MARGIN),
                    f"smallest_margin_{lab}": float(max(abs(lo90), abs(hi90)))})
    rows.append(row)
T = pd.DataFrame(rows)
T.to_csv(os.path.join(OUT, "equivalence_table.csv"), index=False)
json.dump({"margin": MARGIN, "B": B, "rows": json.loads(T.to_json(orient="records"))},
          open(os.path.join(OUT, "equivalence.json"), "w"), indent=1)

# ---- (b) entity-level permutation test for category differences in observed ratings


def norm(s):
    s = str(s).replace("ı", "i").replace("İ", "I")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return s.replace("pastenesi", "pastanesi").strip()


t = pd.read_excel(os.path.join(ROOT, "data/raw/TripAdvisor_rating_counts.xlsx"))
t.columns = ["Attraction", "s5", "s4", "s3", "s2", "s1"]
d["key"] = d.Attraction_Name.map(norm)
t["key"] = t.Attraction.map(norm)
t["cat"] = t.key.map(d.groupby("key").Attraction_Category.first())
t["N"] = t[["s1", "s2", "s3", "s4", "s5"]].sum(axis=1)
t["neg"] = (t.s1 + t.s2) / t.N
t["net"] = (t.s4 + t.s5 - t.s1 - t.s2) / t.N
cats = t.cat.values


def between(v, c, w=None):
    w = np.ones(len(v)) if w is None else w
    m = np.average(v, weights=w)
    return sum(w[c == k].sum() * (np.average(v[c == k], weights=w[c == k]) - m) ** 2 for k in np.unique(c)) / w.sum()


res = {"n_entities": int(len(t)), "category_sizes": t.cat.value_counts().to_dict(), "n_perm": 20000}
prng = np.random.default_rng(1)
for var in ["neg", "net"]:
    for wname, w in [("unweighted", None), ("weighted_by_ratings", t.N.values.astype(float))]:
        obs = between(t[var].values, cats, w)
        perm = np.array([between(t[var].values, prng.permutation(cats), w) for _ in range(20000)])
        res[f"{var}_{wname}"] = {"stat": float(obs), "p": float((1 + (perm >= obs).sum()) / (1 + len(perm)))}
json.dump(res, open(os.path.join(OUT, "category_permutation.json"), "w"), indent=1)
pd.set_option("display.width", 250)
print(T.round(3).to_string())
print(json.dumps(res, indent=1))

# ---- (c) DistilBERT drop on the 508 reviews never used for configuration selection (validation reviews removed)
val_ids = set(json.load(open(os.path.join(ROOT, "data/processed/split_main.json")))["val_ids"])
m = np.array([i not in val_ids for i in Z["ids"]])
a, b, y = Z["random_5fold__DistilBERT"][m], Z["leave_one_attraction_out__DistilBERT"][m], Y[m]
rr = np.random.default_rng(0)
dd = []
for _ in range(B):
    i = rr.integers(0, len(y), len(y)); dd.append(mf1(y[i], a[i]) - mf1(y[i], b[i]))
json.dump({"n": int(m.sum()), "random": mf1(y, a), "loeo": mf1(y, b), "drop": mf1(y, a) - mf1(y, b),
           "ci95": [float(v) for v in np.percentile(dd, [2.5, 97.5])], "ci90": [float(v) for v in np.percentile(dd, [5, 95])]},
          open(os.path.join(OUT, "distilbert_drop_508.json"), "w"), indent=1)
