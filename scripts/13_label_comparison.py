"""
Step 13 - Compare star-derived, lexicon (VADER) and model (DistilBERT, TF-IDF) labels with the blind human audit labels.

Inputs (no review texts needed):
  annotation/audit_labels_blind.csv                      item key + both annotators' text-only labels
  results/label_comparison/vader_compound_audited.csv    VADER compound score for each audited review
  results/generalisation/oof_predictions.npz             pooled out-of-fold predictions
  data/processed/split_main.json                         main split (to exclude validation reviews)
Outputs:
  results/label_comparison/label_comparison.json          Table 2, Fig. 1, Tables S16, S16b, S17
"""
import json, os
import numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score, recall_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results/label_comparison"); os.makedirs(OUT, exist_ok=True)
LABS = ["Negative", "Neutral", "Positive"]; INV = {0: "Negative", 1: "Neutral", 2: "Positive"}

a = pd.read_csv(os.path.join(ROOT, "annotation/audit_labels_blind.csv"))
main = a[(a.Manual_Override == 0) & (a.Review_ID != 420)].copy()          # 141 random audit reviews
assert (main.Rating_Label == main.Final_Sentiment).all()                   # raw == final labels here
vc = pd.read_csv(os.path.join(OUT, "vader_compound_audited.csv")).set_index("Review_ID").vader_compound
band = lambda c, lo, hi: "Positive" if c >= hi else ("Negative" if c <= lo else "Neutral")
main["VADER_standard"] = [band(vc[r], -0.05, 0.05) for r in main.Review_ID]
main["VADER_tuned"] = [band(vc[r], -0.10, 0.70) for r in main.Review_ID]     # band selected on the validation set
z = np.load(os.path.join(ROOT, "results/generalisation/oof_predictions.npz"))
idx = {int(r): i for i, r in enumerate(z["ids"])}
for col, key in [("DistilBERT_CV", "random_5fold__DistilBERT"), ("DistilBERT_LOEO", "leave_one_attraction_out__DistilBERT"),
                 ("TFIDF_LR_CV", "random_5fold__TFIDF-LR")]:
    main[col] = [INV[int(z[key][idx[int(r)]])] for r in main.Review_ID]
main["Star"] = main.Rating_Label
METHODS = ["Star", "DistilBERT_CV", "DistilBERT_LOEO", "TFIDF_LR_CV", "VADER_tuned", "VADER_standard"]

def evaluate(df, ref, B=2000, seed=20261115):
    rng = np.random.default_rng(seed); y = df[ref].values; n = len(df); out = {}
    boots = {m: [] for m in METHODS}
    for _ in range(B):
        i = rng.integers(0, n, n)
        for m in METHODS:
            boots[m].append(f1_score(y[i], df[m].values[i], labels=LABS, average="macro", zero_division=0))
    for m in METHODS:
        p = df[m].values
        out[m] = {"n": n, "agreement": round(float((y == p).mean()), 3), "kappa": round(float(cohen_kappa_score(y, p)), 3),
                  "macro_f1": round(float(f1_score(y, p, labels=LABS, average="macro", zero_division=0)), 3),
                  "macro_f1_ci": [round(float(np.percentile(boots[m], q)), 3) for q in (2.5, 97.5)],
                  "recall_neg_neu_pos": [round(float(recall_score(y, p, labels=[l], average="macro", zero_division=0)), 3) for l in LABS],
                  "confusion_rows_human_cols_method": confusion_matrix(y, p, labels=LABS).tolist()}
    diffs = {}
    for x, w in [("VADER_standard", "Star"), ("DistilBERT_CV", "Star"), ("DistilBERT_CV", "VADER_standard")]:
        d = np.array(boots[x]) - np.array(boots[w])
        diffs[f"{x}_minus_{w}"] = [round(float(d.mean()), 3)] + [round(float(np.percentile(d, q)), 3) for q in (2.5, 97.5)]
    out["paired_macro_f1_differences_mean_ci"] = diffs
    return out

cons = main[main.Annotator_A == main.Annotator_B].copy(); cons["Consensus"] = cons.Annotator_A
split = json.load(open(os.path.join(ROOT, "data/processed/split_main.json")))
val = set(int(r) for r in split["val_ids"]) if isinstance(split.get("val_ids"), list) else set()
res = {
    "accounting": {"selected": len(a), "management_reply_excluded": int((a.Review_ID == 420).sum()),
                   "overrides_assessed_separately": int(a.Manual_Override.sum()), "main_audit": len(main),
                   "consensus": len(cons), "audited_in_validation_set": int(main.Review_ID.isin(val).sum()),
                   "validation_in_consensus": int(cons.Review_ID.isin(val).sum()),
                   "consensus_excluding_validation": int((~cons.Review_ID.isin(val)).sum())},
    "annotators_A_vs_B": {"agreement": round(float((main.Annotator_A == main.Annotator_B).mean()), 3),
                          "kappa": round(float(cohen_kappa_score(main.Annotator_A, main.Annotator_B)), 3)},
    "vs_consensus": evaluate(cons, "Consensus"),
    "vs_annotator_A_141": evaluate(main, "Annotator_A"),
    "vs_annotator_B_141": evaluate(main, "Annotator_B"),
}
if val:
    cv = cons[~cons.Review_ID.isin(val)]
    res["vs_consensus_excluding_validation"] = {m: {"n": len(cv), "kappa": round(float(cohen_kappa_score(cv.Consensus, cv[m])), 3),
                                                    "macro_f1": round(float(f1_score(cv.Consensus, cv[m], labels=LABS, average="macro", zero_division=0)), 3)} for m in METHODS}
json.dump(res, open(os.path.join(OUT, "label_comparison.json"), "w"), indent=1)
print(json.dumps(res["accounting"]), json.dumps(res["annotators_A_vs_B"]))
for m in METHODS:
    r = res["vs_consensus"][m]; print(f"{m:16s} kappa={r['kappa']:.2f} macroF1={r['macro_f1']:.2f} {r['macro_f1_ci']}")
