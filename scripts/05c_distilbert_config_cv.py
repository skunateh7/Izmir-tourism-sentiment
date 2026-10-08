"""
Step 5c — DistilBERT: all four candidate configurations through random 5-fold CV and leave-one-entity-out (LOEO).

Answers the review question about configuration selection: the DistilBERT configuration used in step 5b was chosen
on a validation split that contained reviews from every entity, including entities later held out in LOEO.
Here every configuration (lr in {2e-5, 5e-5} x class weighting {none, balanced}) is run through exactly the folds,
inner validation splits, seeds and training procedure of step 5b. For each outer fold we also save the inner
validation macro-F1 (computed only on reviews from the TRAINING entities of that fold), which allows
entity-nested configuration selection: in each outer fold, pick the configuration with the best inner validation
score, then score that fold's held-out entity.

GPU (Colab) only; 4 configs x 19 folds x 3 seeds = 228 fine-tunes, roughly 4-5 h on a T4, less on L4/A100.
Every finished (config, fold) is saved to --out-dir, so after a disconnect simply run it again and it resumes.
LOEO folds run first. Then run the summary locally:  python scripts/05c_distilbert_config_cv.py --summarise
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.metrics import full_metrics  # noqa: E402
from src.preprocess import LABEL2ID  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="distilbert-base-uncased")
ap.add_argument("--out-dir", default=os.path.join(ROOT, "results/transformer_config_cv"))
ap.add_argument("--max-len", type=int, default=256)
ap.add_argument("--epochs", type=int, default=6)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--summarise", action="store_true", help="no training: summarise saved folds")
args = ap.parse_args()

SEEDS = [42, 7, 13]
CONFIGS = [(2e-5, False), (2e-5, True), (5e-5, False), (5e-5, True)]
cname = lambda lr, cw: f"lr={lr:g},cw={'bal' if cw else 'none'}"
os.makedirs(args.out_dir, exist_ok=True)

d = pd.read_csv(os.path.join(ROOT, "data/processed/reviews_publication_final.csv"))
Y = d.Final_Sentiment.map(LABEL2ID).values
TEXT = d.Original_Review_Text.astype(str).tolist()
ENT = sorted(d.Attraction_Name.unique())

folds = []
for k, a in enumerate(ENT):
    te = np.where(d.Attraction_Name == a)[0]; tr = np.where(d.Attraction_Name != a)[0]
    folds.append(("leave_one_attraction_out", k, tr, te, a))
for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(d, Y)):
    folds.append(("random_5fold", k, tr, te, f"fold{k+1}"))


def fpath(lr, cw, scheme, k):
    return os.path.join(args.out_dir, f"{cname(lr, cw)}__{scheme}__{k:02d}.npz")


if not args.summarise:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", dev, flush=True)
    if dev.type != "cuda":
        print("WARNING: no GPU — this will take days. In Colab: Runtime > Change runtime type > GPU.", flush=True)
    TOK = AutoTokenizer.from_pretrained(args.model)

    def encode(texts):
        e = TOK(texts, truncation=True, max_length=args.max_len, padding="max_length", return_tensors="pt")
        return e["input_ids"], e["attention_mask"]

    @torch.no_grad()
    def predict(mdl, ids, mask):
        mdl.eval()
        return torch.cat([torch.softmax(mdl(input_ids=ids[i:i+64].to(dev), attention_mask=mask[i:i+64].to(dev)).logits, -1).cpu()
                          for i in range(0, len(ids), 64)]).numpy()

    def train_eval(tr_i, va_i, te, seed, LR, CW):
        """Identical to steps 5/5b: AdamW, 10% warm-up + linear decay, grad-clip 1.0, keep best val macro-F1 epoch."""
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        mdl = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3).to(dev)
        Xtr, Xva, Xte = (encode([TEXT[i] for i in idx]) for idx in (tr_i, va_i, te))
        ytr = Y[tr_i]
        dl = DataLoader(TensorDataset(*Xtr, torch.tensor(ytr)), batch_size=args.batch, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
        w = torch.tensor(len(ytr) / (3 * np.bincount(ytr, minlength=3)), dtype=torch.float).to(dev) if CW else None
        lossf = torch.nn.CrossEntropyLoss(weight=w)
        opt = torch.optim.AdamW(mdl.parameters(), lr=LR, weight_decay=0.01)
        total = len(dl) * args.epochs
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / (0.1 * total), max(0.0, (total - s) / (0.9 * total))))
        best, best_state = -1, None
        for _ in range(args.epochs):
            mdl.train()
            for ids, mask, lab in dl:
                opt.zero_grad()
                lossf(mdl(input_ids=ids.to(dev), attention_mask=mask.to(dev)).logits, lab.to(dev)).backward()
                torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
                opt.step(); sched.step()
            f = full_metrics(Y[va_i], predict(mdl, *Xva).argmax(1))["macro_f1"]
            if f > best:
                best, best_state = f, {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
        mdl.load_state_dict(best_state)
        p = predict(mdl, *Xte)
        del mdl; torch.cuda.empty_cache()
        return p, best

    jobs = [(lr, cw, f) for f in folds for (lr, cw) in CONFIGS]  # LOEO folds first (folds list starts with LOEO)
    t0, todo = time.time(), sum(not os.path.exists(fpath(lr, cw, f[0], f[1])) for lr, cw, f in jobs)
    print(f"{len(jobs) - todo}/{len(jobs)} (config, fold) jobs already done; {todo} to go", flush=True)
    done_now = 0
    for lr, cw, (scheme, k, tr, te, name) in jobs:
        fp = fpath(lr, cw, scheme, k)
        if os.path.exists(fp):
            continue
        tr_i, va_i = train_test_split(tr, test_size=0.15, stratify=Y[tr], random_state=42)  # same as steps 5b/6
        out = [train_eval(tr_i, va_i, te, s, lr, cw) for s in SEEDS]
        P = np.mean([o[0] for o in out], 0)
        np.savez(fp, te=te, prob=P, val_f1=np.array([o[1] for o in out]))
        done_now += 1
        el = (time.time() - t0) / 60
        print(f"[{len(jobs) - todo + done_now}/{len(jobs)}] {cname(lr, cw)} {scheme} {name}: "
              f"acc={(P.argmax(1) == Y[te]).mean():.3f} val={np.mean([o[1] for o in out]):.3f} "
              f"({el:.0f} min, ~{el / done_now * (todo - done_now):.0f} min left)", flush=True)
    print("All jobs finished. Download the folder and send it back (or run --summarise).", flush=True)
    sys.exit(0)

# =============================== summary (CPU)
from src.metrics import bootstrap_ci, paired_bootstrap_diff  # noqa: E402

G = os.path.join(ROOT, "results/generalisation")
missing = [(c, f[0], f[1]) for c in CONFIGS for f in folds if not os.path.exists(fpath(*c, f[0], f[1]))]
if missing:
    sys.exit(f"{len(missing)} (config, fold) files missing, e.g. {missing[:3]}")
SCH = ["random_5fold", "leave_one_attraction_out"]
oof, val = {}, {}
for c in CONFIGS:
    for s in SCH:
        o = np.full(len(d), -1); v = {}
        for sc, k, tr, te, name in folds:
            if sc != s:
                continue
            z = np.load(fpath(*c, s, k)); o[z["te"]] = z["prob"].argmax(1); v[k] = float(z["val_f1"].mean())
        oof[(c, s)], val[(c, s)] = o, v
# entity-nested (and fold-nested) selection: per outer fold, the config with best inner validation score
for s in SCH:
    o = np.full(len(d), -1); chosen = {}
    for sc, k, tr, te, name in folds:
        if sc != s:
            continue
        best = max(CONFIGS, key=lambda c: val[(c, s)][k])
        chosen[name] = cname(*best); o[te] = oof[(best, s)][te]
    oof[("nested", s)] = o; val[("nested", s)] = chosen

rows = []
for c in CONFIGS + ["nested"]:
    lab = "nested selection" if c == "nested" else cname(*c)
    r = {"config": lab}
    for s in SCH:
        r[f"{s}_macro_f1"] = full_metrics(Y, oof[(c, s)])["macro_f1"]
        r[f"{s}_ci"] = bootstrap_ci(Y, oof[(c, s)])
    dd = paired_bootstrap_diff(Y, oof[(c, "random_5fold")], oof[(c, "leave_one_attraction_out")])
    r["drop_random_minus_loeo"] = {"diff": r["random_5fold_macro_f1"] - r["leave_one_attraction_out_macro_f1"], **dd}
    rows.append(r)
# per-entity LOEO accuracy / macro-F1 by configuration
per = []
for a in ENT:
    i = np.where(d.Attraction_Name == a)[0]
    rr = {"entity": a, "n": len(i)}
    for c in CONFIGS + ["nested"]:
        lab = "nested" if c == "nested" else cname(*c)
        p = oof[(c, "leave_one_attraction_out")][i]
        rr[f"acc__{lab}"] = float((p == Y[i]).mean())
        rr[f"f1__{lab}"] = full_metrics(Y[i], p)["macro_f1"] if len(i) >= 20 else np.nan
    per.append(rr)
PE = pd.DataFrame(per)
accs = PE[[c for c in PE.columns if c.startswith("acc__lr")]]
PE["acc_range_across_configs"] = accs.max(1) - accs.min(1)
OUT = os.path.join(ROOT, "results/generalisation")
PE.round(4).to_csv(os.path.join(OUT, "distilbert_configs_loeo_per_entity.csv"), index=False)
# nested DistilBERT vs nested TF-IDF (step 14)
cmp = {}
nt = os.path.join(G, "nested_tfidf_oof.npz")
if os.path.exists(nt):
    Z = np.load(nt)
    for s in SCH:
        for fam in ["TFIDF-LR", "TFIDF-SVM"]:
            kk = f"{fam}__{s}"
            if kk in Z.files:
                cmp[f"{s}: nested DistilBERT - nested {fam}"] = {
                    "diff": full_metrics(Y, oof[("nested", s)])["macro_f1"] - full_metrics(Y, Z[kk])["macro_f1"],
                    **paired_bootstrap_diff(Y, oof[("nested", s)], Z[kk])}
res = {"configs": rows, "nested_choices": {s: val[("nested", s)] for s in SCH},
       "nested_distilbert_vs_nested_tfidf": cmp,
       "per_entity_acc_range_across_configs": {"median": float(PE.acc_range_across_configs.median()),
                                               "max": float(PE.acc_range_across_configs.max())},
       "note": "inner validation = 15% stratified split of the outer-fold training reviews (training entities only); "
               "3 seeds per config; probabilities averaged over seeds"}
json.dump(res, open(os.path.join(OUT, "distilbert_configs_cv.json"), "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
print(PE.round(3).to_string(index=False))
