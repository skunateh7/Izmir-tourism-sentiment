"""
Step 5 — DistilBERT fine-tuning on the SAME locked split (run on a GPU machine or Google Colab).

    pip install torch "transformers<5" accelerate scikit-learn pandas
    python scripts/05_transformer_distilbert.py            # full grid (lr × class weighting × 5 seeds)
    python scripts/05_transformer_distilbert.py --quick    # lr=2e-5, class-weighted, 5 seeds
    python scripts/05_transformer_distilbert.py --smoke-test   # tiny random model, no download (CI check)
    python scripts/05_transformer_distilbert.py --export-app-model   # one model (seed 42) for the Streamlit app

Outputs results/transformer/{all_runs.csv,test_predictions.json}. Scripts 07/10 merge them
automatically into the statistics, tables and figures when present.
Model selection uses validation macro-F1 only. The corpus is 100% English (see audit), so an
English model (distilbert-base-uncased) is used rather than a multilingual one.
"""
import argparse
import itertools
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.metrics import full_metrics  # noqa: E402
from src.preprocess import LABEL2ID  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="distilbert-base-uncased")
ap.add_argument("--quick", action="store_true")
ap.add_argument("--smoke-test", action="store_true")
ap.add_argument("--export-app-model", action="store_true",
                help="train only the selected config (lr=2e-5, class-weighted, seed 42) and save it to app/models/distilbert")
ap.add_argument("--max-len", type=int, default=256)
ap.add_argument("--epochs", type=int, default=6)
ap.add_argument("--batch", type=int, default=16)
args = ap.parse_args()

SEEDS = [42, 7, 13, 99, 2024]
GRID = [(2e-5, True)] if args.quick else list(itertools.product([2e-5, 5e-5], [False, True]))
if args.export_app_model:
    SEEDS, GRID = [42], [(2e-5, True)]
if args.smoke_test:
    SEEDS, GRID, args.epochs, args.max_len = [42], [(5e-4, True)], 2, 64
OUT = os.path.join(ROOT, "results/transformer_smoketest" if args.smoke_test else
                   "results/transformer_app_model" if args.export_app_model else "results/transformer")
os.makedirs(OUT, exist_ok=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

d = pd.read_csv(os.path.join(ROOT, "data/processed/reviews_publication_final.csv"))
sp = json.load(open(os.path.join(ROOT, "data/processed/split_main.json")))
part = {k: d[d.Review_ID.isin(sp[f"{k}_ids"])].sort_values("Review_ID").reset_index(drop=True)
        for k in ["train", "val", "test"]}
y = {k: part[k].Final_Sentiment.map(LABEL2ID).values for k in part}
# Transformers receive the ORIGINAL text (their tokenizer handles casing/punctuation/negation).
text = {k: part[k].Original_Review_Text.astype(str).tolist() for k in part}


def load_tokenizer_and_model(seed):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    if not args.smoke_test:
        tok = AutoTokenizer.from_pretrained(args.model)
        mdl = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3)
        return tok, mdl
    # --- offline smoke test: word-level tokenizer + tiny randomly initialised DistilBERT
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import DistilBertConfig, DistilBertForSequenceClassification, PreTrainedTokenizerFast
    t = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    t.train_from_iterator(text["train"], trainers.WordLevelTrainer(special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"]))
    tok = PreTrainedTokenizerFast(tokenizer_object=t, pad_token="[PAD]", unk_token="[UNK]",
                                  cls_token="[CLS]", sep_token="[SEP]")
    cfg = DistilBertConfig(vocab_size=len(tok), dim=64, hidden_dim=128, n_layers=2, n_heads=2,
                           max_position_embeddings=args.max_len, num_labels=3)
    return tok, DistilBertForSequenceClassification(cfg)


def encode(tok, texts):
    e = tok(texts, truncation=True, max_length=args.max_len, padding="max_length", return_tensors="pt")
    return e["input_ids"], e["attention_mask"]


@torch.no_grad()
def predict(mdl, ids, mask):
    mdl.eval()
    out = []
    for i in range(0, len(ids), 64):
        out.append(torch.softmax(mdl(input_ids=ids[i:i+64].to(dev), attention_mask=mask[i:i+64].to(dev)).logits, -1).cpu())
    return torch.cat(out).numpy()


def run(lr, cw, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    tok, mdl = load_tokenizer_and_model(seed)
    mdl.to(dev)
    enc = {k: encode(tok, text[k]) for k in text}
    dl = DataLoader(TensorDataset(*enc["train"], torch.tensor(y["train"])), batch_size=args.batch, shuffle=True,
                    generator=torch.Generator().manual_seed(seed))
    w = None
    if cw:
        cnt = np.bincount(y["train"], minlength=3)
        w = torch.tensor(len(y["train"]) / (3 * cnt), dtype=torch.float).to(dev)
    lossf = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=0.01)
    total = len(dl) * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / (0.1 * total), max(0.0, (total - s) / (0.9 * total))))
    best, best_state, best_ep, hist = -1, None, 0, []
    for ep in range(args.epochs):
        mdl.train()
        for ids, mask, lab in dl:
            opt.zero_grad()
            loss = lossf(mdl(input_ids=ids.to(dev), attention_mask=mask.to(dev)).logits, lab.to(dev))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
            opt.step(); sched.step()
        pv = predict(mdl, *enc["val"]).argmax(1)
        f = full_metrics(y["val"], pv)["macro_f1"]
        hist.append({"epoch": ep + 1, "val_macro_f1": f})
        if f > best:
            best, best_ep = f, ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
    mdl.load_state_dict(best_state)
    if args.export_app_model:  # weights for the Streamlit app (see app/README.md)
        app_dir = os.path.join(ROOT, "app/models/distilbert")
        mdl.config.id2label = {0: "Negative", 1: "Neutral", 2: "Positive"}
        mdl.config.label2id = {"Negative": 0, "Neutral": 1, "Positive": 2}
        mdl.save_pretrained(app_dir); tok.save_pretrained(app_dir)
        print("saved app model ->", app_dir)
    return predict(mdl, *enc["val"]), predict(mdl, *enc["test"]), best_ep, hist


runs, preds = [], {"test_ids": part["test"].Review_ID.tolist(), "y_test": y["test"].tolist()}
t0 = time.time()
for lr, cw in GRID:
    cfg = f"lr={lr:g},cw={'bal' if cw else 'none'}"
    probs = []
    for s in SEEDS:
        pv, pt, be, hist = run(lr, cw, s)
        probs.append(pt)
        mv, mt = full_metrics(y["val"], pv.argmax(1)), full_metrics(y["test"], pt.argmax(1))
        runs.append(dict(family="DistilBERT", config=cfg, seed=s, val_macro_f1=mv["macro_f1"], val_acc=mv["accuracy"],
                         test_macro_f1=mt["macro_f1"], test_acc=mt["accuracy"], test_weighted_f1=mt["weighted_f1"],
                         test_macro_p=mt["macro_precision"], test_macro_r=mt["macro_recall"],
                         **{f"test_f1_{c}": mt["per_class"][c]["f1"] for c in mt["per_class"]},
                         **{f"test_recall_{c}": mt["per_class"][c]["recall"] for c in mt["per_class"]},
                         best_epoch=be, epochs_run=len(hist)))
        preds[f"DistilBERT|{cfg}|seed{s}"] = {"pred": pt.argmax(1).tolist(), "prob": np.round(pt, 5).tolist()}
        print(f"DistilBERT {cfg} seed={s} val={mv['macro_f1']:.3f} test={mt['macro_f1']:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    ens = np.mean(probs, 0)
    preds[f"DistilBERT|{cfg}|ensemble"] = {"pred": ens.argmax(1).tolist(), "prob": np.round(ens, 5).tolist()}

pd.DataFrame(runs).to_csv(os.path.join(OUT, "all_runs.csv"), index=False)
json.dump(preds, open(os.path.join(OUT, "test_predictions.json"), "w"))
json.dump({"model": "smoke-test" if args.smoke_test else args.model, "max_len": args.max_len, "epochs": args.epochs,
           "batch": args.batch, "grid": [list(g) for g in GRID], "seeds": SEEDS, "device": str(dev),
           "runtime_seconds": round(time.time() - t0)}, open(os.path.join(OUT, "meta.json"), "w"), indent=1)
print("DONE ->", OUT)
