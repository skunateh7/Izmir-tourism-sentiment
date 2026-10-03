"""
Step 7b — Higher-powered comparisons on pooled out-of-fold predictions (N = 598).
  * pairwise exact McNemar (Holm) + paired-bootstrap macro-F1 differences, random 5-fold CV
  * per model: paired-bootstrap CI for the drop random-CV -> leave-one-attraction-out
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.metrics import full_metrics, mcnemar_test, paired_bootstrap_diff  # noqa: E402

OUT = os.path.join(ROOT, "results/generalisation")
Z = np.load(os.path.join(OUT, "oof_predictions.npz"))
y = Z["y"]
fams = [k.split("__")[1] for k in Z.files if k.startswith("random_5fold__")]
rows = []
for a, b in itertools.combinations(fams, 2):
    pa, pb = Z[f"random_5fold__{a}"], Z[f"random_5fold__{b}"]
    mc, bd = mcnemar_test(y, pa, pb), paired_bootstrap_diff(y, pa, pb)
    rows.append(dict(A=a, B=b, macroF1_diff=full_metrics(y, pa)["macro_f1"] - full_metrics(y, pb)["macro_f1"],
                     diff_ci_lo=bd["diff_ci95"][0], diff_ci_hi=bd["diff_ci95"][1],
                     A_only_correct=mc["a_only_correct"], B_only_correct=mc["b_only_correct"], mcnemar_p=mc["p_value_exact"]))
P = pd.DataFrame(rows)
P["mcnemar_p_holm"] = multipletests(P.mcnemar_p, method="holm")[1]
P.to_csv(os.path.join(OUT, "pairwise_tests_cv.csv"), index=False)

drop = []
for f in fams:
    pr, pl = Z[f"random_5fold__{f}"], Z[f"leave_one_attraction_out__{f}"]
    bd = paired_bootstrap_diff(y, pr, pl)
    drop.append(dict(family=f, random_macro_f1=full_metrics(y, pr)["macro_f1"], loao_macro_f1=full_metrics(y, pl)["macro_f1"],
                     drop=full_metrics(y, pr)["macro_f1"] - full_metrics(y, pl)["macro_f1"],
                     drop_ci_lo=bd["diff_ci95"][0], drop_ci_hi=bd["diff_ci95"][1],
                     mcnemar_p=mcnemar_test(y, pr, pl)["p_value_exact"]))
D = pd.DataFrame(drop)
D.to_csv(os.path.join(OUT, "random_vs_loao_drop.csv"), index=False)
print(P.round(4).to_string()); print(D.round(4).to_string())
