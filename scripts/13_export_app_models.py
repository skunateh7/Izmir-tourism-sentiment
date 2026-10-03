"""
Step 13 - Export the lightweight models used by the Streamlit app (CPU, seconds).

Refits the validation-selected TF-IDF + LR and TF-IDF + SVM configurations on the locked
training split (exactly as in Step 4), checks that their test macro-F1 reproduces the
manuscript (0.686 and 0.705), and saves them with the validation-tuned VADER band to
app/models/. The saved files contain the fitted vocabulary and weights only - no review text.
"""
import json, os, sys
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC


def tfidf_model(kind, C, ngram, class_weight):  # identical to src/models.py (avoids importing TensorFlow)
    vec = TfidfVectorizer(ngram_range=ngram, min_df=1, sublinear_tf=True)
    clf = (LogisticRegression(C=C, max_iter=5000, class_weight=class_weight)
           if kind == "lr" else LinearSVC(C=C, class_weight=class_weight))
    return vec, clf


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.preprocess import LABELS, LABEL2ID, clean_text  # noqa: E402

d = pd.read_csv(os.path.join(ROOT, "data/processed/reviews_publication_final.csv"))
d["Processed_Text"] = d.Processed_Text.fillna("")
same = (d.Original_Review_Text.map(clean_text) == d.Processed_Text).mean()
print(f"clean_text reproduces the stored processed text for {same:.1%} of reviews")
split = json.load(open(os.path.join(ROOT, "data/processed/split_main.json")))
part = {k: d[d.Review_ID.isin(split[k + '_ids'])] for k in ("train", "val", "test")}
final = json.load(open(os.path.join(ROOT, "results/final/final_results.json")))
cfg = final["selected_configs"]
EXPECTED = {"TFIDF-LR": 0.686, "TFIDF-SVM": 0.705}
out = os.path.join(ROOT, "app/models"); os.makedirs(out, exist_ok=True)
meta = {"sklearn_version": sklearn.__version__, "labels": LABELS, "models": {}}
for fam, kind in [("TFIDF-LR", "lr"), ("TFIDF-SVM", "svm")]:
    c = dict(x.split("=") for x in cfg[fam].split(","))
    ng = tuple(int(v) for v in c["ngram"].split("-"))
    vec, clf = tfidf_model(kind, float(c["C"]), ng, None if c["cw"] == "none" else "balanced")
    clf.fit(vec.fit_transform(part["train"].Processed_Text), part["train"].Final_Sentiment.map(LABEL2ID))
    pt = clf.predict(vec.transform(part["test"].Processed_Text))
    f1 = f1_score(part["test"].Final_Sentiment.map(LABEL2ID), pt, average="macro")
    print(f"{fam} [{cfg[fam]}] test macro-F1 = {f1:.3f} (manuscript {EXPECTED[fam]})")
    assert abs(f1 - EXPECTED[fam]) < 0.0015, "does not reproduce the manuscript"
    joblib.dump({"vectorizer": vec, "classifier": clf}, os.path.join(out, f"{fam.lower()}.joblib"), compress=3)
    meta["models"][fam] = {"config": cfg[fam], "test_macro_f1": round(f1, 4)}
lo, hi = [float(v) for v in cfg["VADER"].split("(")[1].rstrip(")").split(",")]
meta["vader_band"] = {"lo": lo, "hi": hi}
json.dump(meta, open(os.path.join(out, "app_models.json"), "w"), indent=1)
print("saved to", out)
