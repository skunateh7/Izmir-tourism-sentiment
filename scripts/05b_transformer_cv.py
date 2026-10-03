"""
Step 5b — DistilBERT in the generalisation experiment (random 5-fold CV + leave-one-attraction-out CV).

Uses EXACTLY the folds, inner validation splits and seeds of scripts/06_generalisation_cv.py and the
training procedure of scripts/05_transformer_distilbert.py, with the configuration that script 05 selected
on validation (read from results/transformer/all_runs.csv). Run on a GPU (Google Colab), ~1-1.5 h.

    python scripts/05b_transformer_cv.py --out-dir /content/drive/MyDrive/izmir_distilbert_cv

Every finished fold is saved to --out-dir, so if Colab disconnects you simply run it again and it resumes.
When all 19 folds exist, the DistilBERT predictions are merged into results/generalisation/
(oof_predictions.npz, cv_comparison.csv, per_attraction_loao.csv, generalisation_results.json).
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.metrics import bootstrap_ci, full_metrics  # noqa: E402
from src.preprocess import LABEL2ID  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="distilbert-base-uncased")
ap.add_argument("--out-dir", default=os.path.join(ROOT, "results/transformer_cv"))
ap.add_argument("--max-len", type=int, default=256)
ap.add_argument("--epochs", type=int, default=6)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--smoke-test", action="store_true", help="tiny random model, 2 folds, no download")
args = ap.parse_args()

SEEDS = [42, 7, 13]  # same as the CNN/BiLSTM generalisation experiment
if args.smoke_test:
    SEEDS, args.epochs, args.max_len = [42], 1, 64
os.makedirs(args.out_dir, exist_ok=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", dev, flush=True)

# ---- selected configuration (validation-selected in script 05)
R = pd.read_csv(os.path.join(ROOT, "results/transformer/all_runs.csv"))
cfg = R.groupby("config").val_macro_f1.mean().sort_values(ascending=False).index[0]
kv = dict(p.split("=") for p in cfg.split(","))
LR, CW = float(kv["lr"]), kv["cw"] == "bal"
if args.smoke_test:
    LR = 5e-4
print(f"Using validation-selected configuration: {cfg}", flush=True)

d = pd.read_csv(os.path.join(ROOT, "data/processed/reviews_publication_final.csv"))
Y = d.Final_Sentiment.map(LABEL2ID).values
TEXT = d.Original_Review_Text.astype(str).tolist()

folds = []
for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(d, Y)):
    folds.append(("random_5fold", k, tr, te, f"fold{k+1}"))
for k, a in enumerate(sorted(d.Attraction_Name.unique())):
    te = np.where(d.Attraction_Name == a)[0]
    tr = np.where(d.Attraction_Name != a)[0]
    folds.append(("leave_one_attraction_out", k, tr, te, a))
if args.smoke_test:
    folds = [folds[0], folds[5]]


def load(train_texts):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    if not args.smoke_test:
        return (AutoTokenizer.from_pretrained(args.model),
                AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3))
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import DistilBertConfig, DistilBertForSequenceClassification, PreTrainedTokenizerFast
    t = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    t.train_from_iterator(train_texts, trainers.WordLevelTrainer(special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"]))
    tok = PreTrainedTokenizerFast(tokenizer_object=t, pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]")
    c = DistilBertConfig(vocab_size=len(tok), dim=64, hidden_dim=128, n_layers=2, n_heads=2,
                         max_position_embeddings=args.max_len, num_labels=3)
    return tok, DistilBertForSequenceClassification(c)


def encode(tok, texts):
    e = tok(texts, truncation=True, max_length=args.max_len, padding="max_length", return_tensors="pt")
    return e["input_ids"], e["attention_mask"]


@torch.no_grad()
def predict(mdl, ids, mask):
    mdl.eval()
    return torch.cat([torch.softmax(mdl(input_ids=ids[i:i+64].to(dev), attention_mask=mask[i:i+64].to(dev)).logits, -1).cpu()
                      for i in range(0, len(ids), 64)]).numpy()


def train_eval(tr_i, va_i, te, seed):
    """Identical procedure to script 05: AdamW, 10% warm-up + linear decay, grad-clip 1.0, keep best val macro-F1 epoch."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    tok, mdl = load([TEXT[i] for i in tr_i])
    mdl.to(dev)
    Xtr, Xva, Xte = (encode(tok, [TEXT[i] for i in idx]) for idx in (tr_i, va_i, te))
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
    del mdl
    torch.cuda.empty_cache()
    return p


t0 = time.time()
for n, (scheme, k, tr, te, name) in enumerate(folds, 1):
    f = os.path.join(args.out_dir, f"{scheme}__{k:02d}.npz")
    if os.path.exists(f):
        print(f"[{n:2d}/{len(folds)}] {scheme} {name}: already done, skipping", flush=True)
        continue
    tr_i, va_i = train_test_split(tr, test_size=0.15, stratify=Y[tr], random_state=42)  # same as script 06
    P = np.mean([train_eval(tr_i, va_i, te, s) for s in SEEDS], 0)
    np.savez(f, te=te, prob=P)
    acc = float((P.argmax(1) == Y[te]).mean())
    print(f"[{n:2d}/{len(folds)}] {scheme} {name}: n_test={len(te)} acc={acc:.3f}  ({(time.time()-t0)/60:.1f} min)", flush=True)

# ---- merge when complete
done = {s: {} for s in ["random_5fold", "leave_one_attraction_out"]}
for scheme, k, tr, te, name in folds:
    f = os.path.join(args.out_dir, f"{scheme}__{k:02d}.npz")
    if os.path.exists(f):
        done[scheme][k] = np.load(f)
missing = len(folds) - sum(len(v) for v in done.values())
if missing:
    sys.exit(f"{missing} folds still missing — run the script again to resume.")
if args.smoke_test:
    print("Smoke test finished (no merge)."); sys.exit(0)

G = os.path.join(ROOT, "results/generalisation")
Z = dict(np.load(os.path.join(G, "oof_predictions.npz")))
rows, res = [], json.load(open(os.path.join(G, "generalisation_results.json")))
for scheme in done:
    oof = np.full(len(d), -1)
    for z in done[scheme].values():
        oof[z["te"]] = z["prob"].argmax(1)
    assert (oof >= 0).all()
    Z[f"{scheme}__DistilBERT"] = oof
    m = full_metrics(Y, oof); m["macro_f1_ci95"] = bootstrap_ci(Y, oof)
    res["schemes"][scheme]["DistilBERT"] = m
    rows.append(dict(scheme=scheme, family="DistilBERT", macro_f1=m["macro_f1"], ci_lo=m["macro_f1_ci95"][0],
                     ci_hi=m["macro_f1_ci95"][1], accuracy=m["accuracy"], f1_Negative=m["per_class"]["Negative"]["f1"],
                     f1_Neutral=m["per_class"]["Neutral"]["f1"], f1_Positive=m["per_class"]["Positive"]["f1"]))
res["selected_configs"]["DistilBERT"] = cfg
np.savez(os.path.join(G, "oof_predictions.npz"), **Z)
json.dump(res, open(os.path.join(G, "generalisation_results.json"), "w"), indent=1)
C = pd.read_csv(os.path.join(G, "cv_comparison.csv"))
C = pd.concat([C[C.family != "DistilBERT"], pd.DataFrame(rows)], ignore_index=True)
C.to_csv(os.path.join(G, "cv_comparison.csv"), index=False)
PA = pd.read_csv(os.path.join(G, "per_attraction_loao.csv"))
PA = PA[PA.family != "DistilBERT"]
add = []
for a, sub in d.groupby("Attraction_Name"):
    i = sub.index.values
    pl, pr = Z["leave_one_attraction_out__DistilBERT"][i], Z["random_5fold__DistilBERT"][i]
    add.append(dict(attraction=a, category=sub.Attraction_Category.iloc[0], n=len(i), family="DistilBERT",
                    loao_acc=float((pl == Y[i]).mean()), random_acc=float((pr == Y[i]).mean()),
                    loao_macro_f1=full_metrics(Y[i], pl)["macro_f1"] if len(i) >= 20 else np.nan))
pd.concat([PA, pd.DataFrame(add)], ignore_index=True).to_csv(os.path.join(G, "per_attraction_loao.csv"), index=False)
print(pd.DataFrame(rows).round(3).to_string())
print("Merged into results/generalisation/. Next: python scripts/07b_cv_statistics.py and scripts/11_figures_tables.py")
