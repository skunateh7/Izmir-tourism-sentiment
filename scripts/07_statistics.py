"""
Step 7 — Final model-comparison table and statistical tests on the locked 90-review test set.

For each family the validation-selected configuration is reported.
  * Neural models: mean ± SD over 5 seeds (primary) and the 5-seed soft-voting ensemble.
  * Deterministic models (VADER, TF-IDF): single run.
Uncertainty: 2,000-sample bootstrap 95% CIs on macro-F1 and accuracy (representative predictions:
deterministic output or seed ensemble). Pairwise comparison: exact McNemar test (Holm-adjusted)
and paired-bootstrap CI for the macro-F1 difference.
"""
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.metrics import bootstrap_ci, full_metrics, mcnemar_test, paired_bootstrap_diff  # noqa: E402
from src.select import load_runs, selected_configs  # noqa: E402

OUT = os.path.join(ROOT, "results/final")
os.makedirs(OUT, exist_ok=True)
R = load_runs()
SEL = selected_configs(R)
P = json.load(open(os.path.join(ROOT, "results/main/test_predictions.json")))
tp = os.path.join(ROOT, "results/transformer/test_predictions.json")
if os.path.exists(tp):
    T = json.load(open(tp))
    assert T["test_ids"] == P["test_ids"], "transformer test IDs differ from locked split"
    P.update({k: v for k, v in T.items() if "|" in k})
y = np.array(P["y_test"])
ORDER = [f for f in ["VADER", "TFIDF-LR", "TFIDF-SVM", "BiLSTM", "CNN", "DistilBERT"] if f in SEL]
NEURAL = {"BiLSTM", "CNN", "DistilBERT"}

rows, rep, details = [], {}, {}
for f in ORDER:
    cfg = SEL[f]
    sub = R[(R.family == f) & (R.config == cfg)]
    key = f"{f}|{cfg}|ensemble" if f in NEURAL else f"{f}|{cfg}"
    rep[f] = np.array(P[key]["pred"])
    m = full_metrics(y, rep[f])
    ci_f1, ci_acc = bootstrap_ci(y, rep[f]), bootstrap_ci(y, rep[f], "accuracy")
    row = dict(Model=f, Config=cfg, n_runs=len(sub), Val_MacroF1=sub.val_macro_f1.mean())
    for col, src in [("Accuracy", "test_acc"), ("Macro_P", "test_macro_p"), ("Macro_R", "test_macro_r"),
                     ("Macro_F1", "test_macro_f1"), ("Weighted_F1", "test_weighted_f1"),
                     ("F1_Negative", "test_f1_Negative"), ("F1_Neutral", "test_f1_Neutral"),
                     ("F1_Positive", "test_f1_Positive"), ("Recall_Neutral", "test_recall_Neutral")]:
        row[col] = sub[src].mean()
        row[col + "_SD"] = sub[src].std(ddof=1) if len(sub) > 1 else 0.0
    row.update(Rep_Accuracy=m["accuracy"], Rep_MacroF1=m["macro_f1"], Rep_MacroF1_CI_lo=ci_f1[0],
               Rep_MacroF1_CI_hi=ci_f1[1], Rep_Acc_CI_lo=ci_acc[0], Rep_Acc_CI_hi=ci_acc[1])
    rows.append(row)
    details[f] = {"config": cfg, "representative_prediction": key, "metrics": m,
                  "macro_f1_ci95": ci_f1, "accuracy_ci95": ci_acc}
main = pd.DataFrame(rows)
main.to_csv(os.path.join(OUT, "table_main_results.csv"), index=False)

best = main.sort_values("Val_MacroF1", ascending=False).iloc[0].Model  # chosen on VALIDATION
pairs = []
for a, b in itertools.combinations(ORDER, 2):
    mc = mcnemar_test(y, rep[a], rep[b])
    bd = paired_bootstrap_diff(y, rep[a], rep[b])
    pairs.append(dict(A=a, B=b, A_only_correct=mc["a_only_correct"], B_only_correct=mc["b_only_correct"],
                      mcnemar_p=mc["p_value_exact"], macroF1_diff_A_minus_B=float(
                          full_metrics(y, rep[a])["macro_f1"] - full_metrics(y, rep[b])["macro_f1"]),
                      diff_ci_lo=bd["diff_ci95"][0], diff_ci_hi=bd["diff_ci95"][1]))
PW = pd.DataFrame(pairs)
PW["mcnemar_p_holm"] = multipletests(PW.mcnemar_p, method="holm")[1]
PW.to_csv(os.path.join(OUT, "table_pairwise_tests.csv"), index=False)

# full configuration grid (appendix)
grid = R.groupby(["family", "config"], sort=False).agg(
    runs=("seed", "count"), val_macro_f1=("val_macro_f1", "mean"), val_sd=("val_macro_f1", "std"),
    test_macro_f1=("test_macro_f1", "mean"), test_sd=("test_macro_f1", "std")).reset_index()
grid["selected"] = [SEL.get(f) == c for f, c in zip(grid.family, grid.config)]
grid.to_csv(os.path.join(OUT, "table_config_grid.csv"), index=False)

json.dump({"selected_configs": SEL, "best_model_by_validation": best, "test_n": int(len(y)),
           "test_distribution": {l: int((y == i).sum()) for i, l in enumerate(["Negative", "Neutral", "Positive"])},
           "details": details}, open(os.path.join(OUT, "final_results.json"), "w"), indent=1)
pd.set_option("display.width", 250)
print(main[["Model", "Config", "Val_MacroF1", "Accuracy", "Macro_F1", "Macro_F1_SD", "Weighted_F1", "F1_Neutral",
            "Rep_MacroF1", "Rep_MacroF1_CI_lo", "Rep_MacroF1_CI_hi"]].round(3).to_string())
print(PW.round(3).to_string())
print("best by validation:", best)
