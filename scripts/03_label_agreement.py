"""
Step 3 — Compute label-audit statistics once both annotation sheets are filled in.

Usage: python scripts/03_label_agreement.py
Reads annotation/annotation_sheet_A.xlsx, annotation_sheet_B.xlsx and the key file.
Writes results/label_audit/label_agreement.json + adjudication_needed.csv
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = os.path.join(ROOT, "annotation")
OUT = os.path.join(ROOT, "results/label_audit")
os.makedirs(OUT, exist_ok=True)
LABS = ["Negative", "Neutral", "Positive"]

key = pd.read_csv(os.path.join(A, "annotation_key_DO_NOT_SHARE.csv"))
a = pd.read_excel(os.path.join(A, "annotation_sheet_A.xlsx")).iloc[:, [0, 2, 3]]
b = pd.read_excel(os.path.join(A, "annotation_sheet_B.xlsx")).iloc[:, [0, 2, 3]]
a.columns = ["Item", "A", "A_conf"]; b.columns = ["Item", "B", "B_conf"]
m = key.merge(a, on="Item").merge(b, on="Item")
m = m.dropna(subset=["A", "B"])
for c in ["A", "B"]:
    m[c] = m[c].str.strip().str.capitalize()


def boot_kappa(x, y, n=2000, seed=0):
    rng = np.random.default_rng(seed); x, y = np.asarray(x), np.asarray(y); ks = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x)); ks.append(cohen_kappa_score(x[i], y[i]))
    return [round(float(np.percentile(ks, 2.5)), 3), round(float(np.percentile(ks, 97.5)), 3)]


res = {"n_items_annotated": int(len(m))}
res["A_vs_B"] = {"percent_agreement": round(float((m.A == m.B).mean() * 100), 1),
                 "cohen_kappa": round(float(cohen_kappa_score(m.A, m.B)), 3),
                 "kappa_95CI": boot_kappa(m.A, m.B),
                 "weighted_kappa_linear": round(float(cohen_kappa_score(m.A, m.B, labels=LABS, weights="linear")), 3)}
agreed = m[m.A == m.B]
for who in ["A", "B"]:
    res[f"{who}_vs_final_label"] = {"percent_agreement": round(float((m[who] == m.Final_Sentiment).mean() * 100), 1),
                                    "cohen_kappa": round(float(cohen_kappa_score(m[who], m.Final_Sentiment)), 3)}
res["consensus_vs_final_label"] = {
    "n_consensus_items": int(len(agreed)),
    "percent_agreement": round(float((agreed.A == agreed.Final_Sentiment).mean() * 100), 1),
    "cohen_kappa": round(float(cohen_kappa_score(agreed.A, agreed.Final_Sentiment)), 3),
    "confusion_rows_final_cols_consensus": confusion_matrix(agreed.Final_Sentiment, agreed.A, labels=LABS).tolist()}
ov = agreed[agreed.Manual_Override == 1]
res["overrides_confirmed_by_consensus"] = f"{int((ov.A == ov.Final_Sentiment).sum())}/{len(ov)}"
res["interpretation_scale"] = "Landis & Koch (1977): <0.20 slight, 0.21-0.40 fair, 0.41-0.60 moderate, 0.61-0.80 substantial, >0.80 almost perfect"

m[m.A != m.B].to_csv(os.path.join(OUT, "adjudication_needed.csv"), index=False)
json.dump(res, open(os.path.join(OUT, "label_agreement.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
