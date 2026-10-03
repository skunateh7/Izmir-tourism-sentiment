"""Helper for script 11: builds the standard-threshold VADER row in the main-table format."""
import json
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metrics import bootstrap_ci, full_metrics  # noqa: E402


def vader_std_row(vs, R):
    P = json.load(open(R("main/test_predictions.json")))
    y = np.array(P["y_test"])
    p = np.array(P[f"VADER|{vs.config}"]["pred"])
    m = full_metrics(y, p)
    lo, hi = bootstrap_ci(y, p)
    row = dict(Model="VADER-std", Config=vs.config, n_runs=1, Val_MacroF1=vs.val_macro_f1,
               Accuracy=m["accuracy"], Accuracy_SD=0, Macro_P=m["macro_precision"], Macro_R=m["macro_recall"],
               Macro_F1=m["macro_f1"], Macro_F1_SD=0, Weighted_F1=m["weighted_f1"],
               F1_Negative=m["per_class"]["Negative"]["f1"], F1_Neutral=m["per_class"]["Neutral"]["f1"],
               F1_Positive=m["per_class"]["Positive"]["f1"], Recall_Neutral=m["per_class"]["Neutral"]["recall"],
               Rep_MacroF1=m["macro_f1"], Rep_MacroF1_CI_lo=lo, Rep_MacroF1_CI_hi=hi)
    return row
