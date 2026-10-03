"""
Step 8 — Ablation study on the main split (validation AND test reported; 5 seeds for CNN).

Each row changes ONE design decision relative to the selected configuration:
  preprocessing : no negation preservation | no apostrophe normalisation (v5 behaviour) | no stopword removal
  CNN structure : kernels {3} | kernels {2,3,4,5}
  training      : class weighting flipped | embedding initialisation flipped
TF-IDF-LR (deterministic) is ablated on the preprocessing choices as a model-agnostic check.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.metrics import full_metrics  # noqa: E402
from src.models import HP, glove_matrix, tfidf_model, train_neural  # noqa: E402
from src.preprocess import LABEL2ID, Vocab, clean_text  # noqa: E402
from src.select import parse_neural, parse_tfidf, selected_configs  # noqa: E402

OUT = os.path.join(ROOT, "results/ablation")
os.makedirs(OUT, exist_ok=True)
SEEDS = [42, 7, 13, 99, 2024]
d = pd.read_csv(os.path.join(ROOT, "data/processed/reviews_publication_final.csv"))
sp = json.load(open(os.path.join(ROOT, "data/processed/split_main.json")))
idx = {k: d.index[d.Review_ID.isin(sp[f"{k}_ids"])].values for k in ["train", "val", "test"]}
Y = d.Final_Sentiment.map(LABEL2ID).values
SEL = selected_configs()
emb0, cw0 = parse_neural(SEL["CNN"])
C, ng, cwl = parse_tfidf(SEL["TFIDF-LR"])

PRE = {"full pipeline": {},
       "no negation preservation": {"keep_negations": False},
       "no apostrophe normalisation (v5)": {"normalise_apostrophes": False},
       "no stopword removal": {"remove_stopwords": False}}
texts = {k: d.Original_Review_Text.apply(lambda t: clean_text(t, **kw)).values for k, kw in PRE.items()}
neg_affected = int(sum(a != b for a, b in zip(texts["full pipeline"], texts["no apostrophe normalisation (v5)"])))

rows = []
# ---- TF-IDF-LR preprocessing ablation
for name in PRE:
    vec, clf = tfidf_model("lr", C, ng, cwl)
    clf.fit(vec.fit_transform(texts[name][idx["train"]]), Y[idx["train"]])
    rv = full_metrics(Y[idx["val"]], clf.predict(vec.transform(texts[name][idx["val"]])))
    rt = full_metrics(Y[idx["test"]], clf.predict(vec.transform(texts[name][idx["test"]])))
    rows.append(dict(model="TFIDF-LR", variant=name, val_mean=rv["macro_f1"], val_sd=0, test_mean=rt["macro_f1"],
                     test_sd=0, test_f1_Neutral=rt["per_class"]["Neutral"]["f1"]))
    print(rows[-1], flush=True)

# ---- CNN ablation
VARIANTS = [("selected configuration", "full pipeline", {}),
            ("no negation preservation", "no negation preservation", {}),
            ("no apostrophe normalisation (v5)", "no apostrophe normalisation (v5)", {}),
            ("no stopword removal", "no stopword removal", {}),
            ("kernels {3} only", "full pipeline", {"kernels": (3,)}),
            ("kernels {2,3,4,5}", "full pipeline", {"kernels": (2, 3, 4, 5)}),
            (f"class weighting {'off' if cw0 else 'on'}", "full pipeline", {"cw": not cw0}),
            (f"embeddings {'random' if emb0 == 'glove' else 'GloVe'}", "full pipeline",
             {"emb": "random" if emb0 == "glove" else "glove"})]
ONLY = os.environ.get("ABL_VARIANT")  # run a single variant (used by the per-process driver below)
CACHE = os.path.join(OUT, "cache"); os.makedirs(CACHE, exist_ok=True)
for vi, (name, pre, kw) in enumerate(VARIANTS):
    cf = os.path.join(CACHE, f"cnn_{vi}.json")
    if os.path.exists(cf):
        rows.append(json.load(open(cf))); continue
    if ONLY is None:  # spawn a fresh process per variant: avoids memory growth across many Keras models
        import subprocess
        subprocess.run([sys.executable, __file__], env={**os.environ, "ABL_VARIANT": str(vi)}, check=True)
        rows.append(json.load(open(cf))); continue
    if int(ONLY) != vi:
        continue
    T = texts[pre]
    vocab = Vocab(HP["max_vocab"]).fit(T[idx["train"]])
    X = {k: vocab.encode(list(T[idx[k]]), HP["max_len"]) for k in idx}
    emb, cw = kw.get("emb", emb0), kw.get("cw", cw0)
    G = glove_matrix(vocab)[0] if emb == "glove" else None
    fv, ft, fn = [], [], []
    for s in SEEDS:
        m, _ = train_neural("cnn", X["train"], Y[idx["train"]], X["val"], Y[idx["val"]], len(vocab), s, G, cw,
                            kernels=kw.get("kernels"))
        fv.append(full_metrics(Y[idx["val"]], m.predict(X["val"], verbose=0).argmax(1))["macro_f1"])
        mt = full_metrics(Y[idx["test"]], m.predict(X["test"], verbose=0).argmax(1))
        ft.append(mt["macro_f1"]); fn.append(mt["per_class"]["Neutral"]["f1"])
    r = dict(model="CNN", variant=name, val_mean=float(np.mean(fv)), val_sd=float(np.std(fv, ddof=1)),
             test_mean=float(np.mean(ft)), test_sd=float(np.std(ft, ddof=1)), test_f1_Neutral=float(np.mean(fn)),
             val_per_seed=[round(float(x), 4) for x in fv], test_per_seed=[round(float(x), 4) for x in ft])
    json.dump(r, open(cf, "w"))
    print(r, flush=True)
if ONLY is not None:
    sys.exit(0)

A = pd.DataFrame(rows)
A.to_csv(os.path.join(OUT, "ablation.csv"), index=False)
json.dump({"selected_cnn": SEL["CNN"], "selected_lr": SEL["TFIDF-LR"], "seeds": SEEDS,
           "reviews_changed_by_apostrophe_fix": neg_affected}, open(os.path.join(OUT, "ablation_meta.json"), "w"), indent=1)
print(A.round(3).to_string())
