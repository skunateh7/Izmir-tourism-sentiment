"""Validation-based configuration selection shared by all downstream scripts."""
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_runs(include_transformer=True):
    R = pd.read_csv(os.path.join(ROOT, "results/main/all_runs.csv"))
    tp = os.path.join(ROOT, "results/transformer/all_runs.csv")
    if include_transformer and os.path.exists(tp):
        R = pd.concat([R, pd.read_csv(tp)], ignore_index=True)
    return R


def selected_configs(R=None):
    """Per family: the configuration with the highest MEAN validation macro-F1 (ties -> first)."""
    R = load_runs() if R is None else R
    g = R.groupby(["family", "config"], sort=False).val_macro_f1.mean().reset_index()
    return {f: sub.sort_values("val_macro_f1", ascending=False, kind="stable").iloc[0].config
            for f, sub in g.groupby("family", sort=False)}


def parse_neural(cfg):
    kv = dict(p.split("=") for p in cfg.split(","))
    return kv["emb"], kv["cw"] == "bal"


def parse_tfidf(cfg):
    kv = dict(p.split("=") for p in cfg.split(","))
    a, b = kv["ngram"].split("-")
    return float(kv["C"]), (int(a), int(b)), ("balanced" if kv["cw"] == "bal" else None)
