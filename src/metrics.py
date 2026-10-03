"""Evaluation metrics, bootstrap confidence intervals and McNemar's test."""
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from statsmodels.stats.contingency_tables import mcnemar

LABELS = ["Negative", "Neutral", "Positive"]


def full_metrics(y, p):
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=[0, 1, 2], zero_division=0)
    return {
        "accuracy": float(accuracy_score(y, p)),
        "macro_precision": float(pr.mean()),
        "macro_recall": float(rc.mean()),
        "macro_f1": float(f1_score(y, p, average="macro", labels=[0, 1, 2], zero_division=0)),
        "weighted_f1": float(f1_score(y, p, average="weighted", labels=[0, 1, 2], zero_division=0)),
        "per_class": {LABELS[i]: {"precision": float(pr[i]), "recall": float(rc[i]),
                                   "f1": float(f1[i]), "support": int(sup[i])} for i in range(3)},
        "confusion_matrix": confusion_matrix(y, p, labels=[0, 1, 2]).tolist(),
    }


def bootstrap_ci(y, p, metric="macro_f1", n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    f = {"macro_f1": lambda a, b: f1_score(a, b, average="macro", labels=[0, 1, 2], zero_division=0),
         "accuracy": accuracy_score}[metric]
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        vals.append(f(y[i], p[i]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def paired_bootstrap_diff(y, pa, pb, n=2000, seed=0):
    """95% CI and one-sided p for macro-F1(A) - macro-F1(B) on the same test items."""
    rng = np.random.default_rng(seed)
    y, pa, pb = map(np.asarray, (y, pa, pb))
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        d.append(f1_score(y[i], pa[i], average="macro", labels=[0, 1, 2], zero_division=0)
                 - f1_score(y[i], pb[i], average="macro", labels=[0, 1, 2], zero_division=0))
    d = np.array(d)
    return {"diff_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_diff_le_0": float((d <= 0).mean())}


def mcnemar_test(y, pa, pb):
    a_ok, b_ok = np.asarray(pa) == np.asarray(y), np.asarray(pb) == np.asarray(y)
    t = [[int((a_ok & b_ok).sum()), int((a_ok & ~b_ok).sum())],
         [int((~a_ok & b_ok).sum()), int((~a_ok & ~b_ok).sum())]]
    r = mcnemar(t, exact=True)
    return {"table": t, "a_only_correct": t[0][1], "b_only_correct": t[1][0],
            "p_value_exact": float(r.pvalue)}
