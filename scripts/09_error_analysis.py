"""
Step 9 — Error analysis.

Part A (objective): using pooled out-of-fold predictions from random 5-fold CV (all 598 reviews),
  error rates by review length, contrast markers (but/however/although...), explicit negation,
  3-star reviews, manual overrides and category — per model family.
Part B (qualitative): 50 misclassified reviews of the validation-selected best model, sampled
  proportionally to its (true -> predicted) error types, exported for manual coding into the taxonomy
  below. After coding (column Error_Type in results/errors/error_coding.csv), rerun to summarise.
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "results/errors")
os.makedirs(OUT, exist_ok=True)
LAB = np.array(["Negative", "Neutral", "Positive"])
TAXONOMY = {
    "mixed_sentiment": "Both praise and complaint; overall verdict depends on weighting of aspects",
    "neutral_boundary": "Mild/lukewarm or 'average' evaluation near the neutral boundary",
    "negation_or_contrast": "Polarity reversed by negation or a contrastive clause ('but', 'however')",
    "implicit_sentiment": "Sentiment implied by facts/events (prices, waiting, incidents) without evaluative words",
    "rating_text_mismatch": "Text polarity clearly differs from the star-derived label",
    "sarcasm_irony": "Literal wording opposite to intended meaning",
    "short_or_uninformative": "Too short / generic to contain decisive cues",
    "domain_vocabulary": "Decisive cue is rare/domain-specific vocabulary (dish names, local terms)",
    "non_standard_language": "Machine-translation artefacts, misspellings or non-standard orthography (e.g. dotless ı)",
    "no_clear_cause": "No identifiable linguistic cause",
}

d = pd.read_csv(os.path.join(ROOT, "data/processed/reviews_publication_final.csv"))
oof = np.load(os.path.join(ROOT, "results/generalisation/oof_predictions.npz"))
Y = oof["y"]
assert (oof["ids"] == d.Review_ID.values).all()
fams = [k.split("__")[1] for k in oof.files if k.startswith("random_5fold__")]
# Best model BY VALIDATION among families that have pooled CV predictions (DistilBERT is main-split only).
from src.select import load_runs  # noqa: E402
_v = load_runs().groupby(["family", "config"]).val_macro_f1.mean().groupby("family").max()
best = _v[[f for f in fams if f in _v.index]].idxmax()

low = d.Original_Review_Text.str.lower().str.replace("’", "'")
feat = pd.DataFrame({
    "length_bin": pd.cut(d.Word_Count, [0, 25, 45, 75, 1000], labels=["<=25", "26-45", "46-75", ">75"]),
    "contrast_marker": low.str.contains(r"\b(?:but|however|although|though|yet|except|despite)\b"),
    "explicit_negation": low.str.contains(r"\b(?:not|no|never|nothing|n't)\b|n't"),
    "three_star": d.Star_Rating == 3,
    "manual_override": d.Manual_Override == 1,
    "category": d.Attraction_Category,
})
rows = []
for f in fams:
    err = oof[f"random_5fold__{f}"] != Y
    for col in feat.columns:
        for lvl, g in feat.groupby(col, observed=True):
            rows.append(dict(model=f, factor=col, level=str(lvl), n=len(g), error_rate=float(err[g.index].mean())))
    rows.append(dict(model=f, factor="overall", level="all", n=len(d), error_rate=float(err.mean())))
F = pd.DataFrame(rows)
F.to_csv(os.path.join(OUT, "error_rates_by_factor.csv"), index=False)

# ---- confusion of the best model (pooled)
p = oof[f"random_5fold__{best}"]
wrong = np.where(p != Y)[0]
pairs = pd.Series([f"{LAB[Y[i]]}->{LAB[p[i]]}" for i in wrong]).value_counts()
rng = np.random.default_rng(2026)
alloc = (pairs / pairs.sum() * 50).round().astype(int)
alloc.iloc[0] += 50 - alloc.sum()
pick = []
for pair, k in alloc.items():
    cand = [i for i in wrong if f"{LAB[Y[i]]}->{LAB[p[i]]}" == pair]
    pick += list(rng.choice(cand, min(k, len(cand)), replace=False))
S = d.loc[pick, ["Review_ID", "Attraction_Name", "Attraction_Category", "Star_Rating", "Final_Sentiment",
                 "Manual_Override", "Word_Count", "Original_Review_Text"]].copy()
S["Predicted"] = LAB[p[pick]]
S["Error_Type"] = ""
S["Coder_Note"] = ""
S.sort_values(["Final_Sentiment", "Predicted"]).to_csv(os.path.join(OUT, "error_sample_for_coding.csv"), index=False)

summary = {"best_model": best, "pooled_errors_best": int(len(wrong)), "error_pairs_best": pairs.to_dict(),
           "sample_size": len(pick), "taxonomy": TAXONOMY}
coded = os.path.join(OUT, "error_coding.csv")
if os.path.exists(coded):
    E = pd.read_csv(coded)
    vc = E.Error_Type.str.split(";").explode().str.strip().value_counts()
    summary["coded_taxonomy_counts"] = vc.to_dict()
    summary["coded_primary_type_counts"] = E.Error_Type.str.split(";").str[0].str.strip().value_counts().to_dict()
json.dump(summary, open(os.path.join(OUT, "error_summary.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in summary.items() if k != "taxonomy"}, indent=1))
print(F[F.model == best].round(3).to_string())
