import os, sys  # noqa: E401
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import APP, LABELS, SENT, csv, layout, setup
from src.preprocess import clean_text

setup("Analyse a review", "🎯")
st.title("Analyse your own review")

MD = os.path.join(APP, "models")
META = json.load(open(os.path.join(MD, "app_models.json")))
G = csv("generalisation", "cv_comparison.csv").set_index(["scheme", "family"])
cvf1 = lambda f: G.loc[("random_5fold", f), "macro_f1"]  # noqa: E731


@st.cache_resource
def tfidf(name):
    import joblib
    return joblib.load(os.path.join(MD, f"{name.lower()}.joblib"))


@st.cache_resource
def vader():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()


def distilbert_source():
    local = os.path.join(MD, "distilbert")
    if os.path.exists(os.path.join(local, "config.json")):
        return local
    try:
        hub = st.secrets.get("DISTILBERT_MODEL", "")
    except Exception:
        hub = ""
    return os.environ.get("DISTILBERT_MODEL", hub) or None


@st.cache_resource
def distilbert(src):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(src)
    mdl = AutoModelForSequenceClassification.from_pretrained(src).eval()
    return tok, mdl


def softmax(z):
    z = np.asarray(z, float); z = np.exp(z - z.max(-1, keepdims=True))
    return z / z.sum(-1, keepdims=True)


def band(key):
    return (META["vader_band"]["lo"], META["vader_band"]["hi"]) if key == "VADER-tuned" else (-0.05, 0.05)


def predict(key, texts):
    """key in TFIDF-SVM, TFIDF-LR, VADER-tuned, VADER, DistilBERT -> (labels, scores, score kind)."""
    if key.startswith("TFIDF"):
        m = tfidf(key)
        X = m["vectorizer"].transform([clean_text(t) for t in texts])
        clf = m["classifier"]
        if hasattr(clf, "predict_proba"):
            return clf.predict(X), clf.predict_proba(X), "Probability"
        return clf.predict(X), softmax(clf.decision_function(X)), "Relative score (softmax of SVM margins, not a probability)"
    if key.startswith("VADER"):
        lo, hi = band(key)
        comp = np.array([vader().polarity_scores(t)["compound"] for t in texts])
        return np.where(comp >= hi, 2, np.where(comp <= lo, 0, 1)), comp, "VADER compound"
    import torch
    tok, mdl = distilbert(distilbert_source())
    with torch.no_grad():
        enc = tok(list(texts), truncation=True, max_length=256, padding=True, return_tensors="pt")
        p = torch.softmax(mdl(**enc).logits, -1).numpy()
    return p.argmax(1), p, "Probability"


def available(module):
    import importlib.util
    return importlib.util.find_spec(module) is not None


HAS_VADER = available("vaderSentiment")
MODELS = {"TF-IDF + SVM": "TFIDF-SVM", "TF-IDF + LR": "TFIDF-LR"}
if HAS_VADER:
    MODELS.update({"VADER (tuned band)": "VADER-tuned", "VADER (standard)": "VADER"})
if distilbert_source() and available("torch") and available("transformers"):
    MODELS = {"DistilBERT (fine-tuned)": "DistilBERT", **MODELS}
PERF = {"TFIDF-SVM": cvf1("TFIDF-SVM"), "TFIDF-LR": cvf1("TFIDF-LR"), "VADER": cvf1("VADER"),
        "DistilBERT": cvf1("DistilBERT"), "VADER-tuned": None}

with st.sidebar:
    model = st.selectbox("Model", list(MODELS))
    key = MODELS[model]
    if PERF.get(key) is not None:
        st.caption(f"Pooled cross-validated macro-F1 in the study: **{PERF[key]:.3f}**")
    if not HAS_VADER:
        st.caption("VADER is hidden because the `vaderSentiment` package is not installed "
                   "(`pip install vaderSentiment`).")
    if distilbert_source() and not (available("torch") and available("transformers")):
        st.caption("A DistilBERT model was found, but `torch` and `transformers` are not installed.")
    elif not distilbert_source():
        st.caption("DistilBERT, the study's best model, is not installed in this deployment. "
                   "See app/README.md to add it.")

st.warning("These models were trained on 418 English TripAdvisor reviews of İzmir tourism entities. They are reliable "
           "for clearly positive or negative text but **not for individual Neutral or mixed reviews**; use them for "
           "aggregate trends, and read mixed reviews yourself.", icon="⚠️")

EX = {"Positive example": "Stunning views from the terrace, friendly staff and the best breakfast we had on the whole trip.",
      "Negative example": "We waited forty minutes for a table, the waiter never brought the menu and the bill was wrong.",
      "Mixed example": "The beach itself is beautiful with clear water, but it was packed and there was litter everywhere.",
      "Lukewarm example": "It was fine. Nothing special, the food was average and the prices were what you would expect."}


def use_example(t):
    st.session_state.review_text = t


st.markdown("**Type or paste your own review** below and press **Analyse review**. "
            "Or click an example to fill the box.")
ec = st.columns(len(EX))
for col, (name, t) in zip(ec, EX.items()):
    col.button(name, on_click=use_example, args=(t,), width="stretch")

with st.form("classify"):
    text = st.text_area("Your review (English)", key="review_text", height=150,
                        placeholder="e.g. The food was delicious but the service was very slow and the prices were too high.")
    compare = st.checkbox("Also show what every model predicts", value=True)
    submitted = st.form_submit_button("Analyse review", type="primary")

if submitted and not text.strip():
    st.info("Please type a review first.")
if submitted and text.strip():
    st.session_state.last = (text, compare)
if "last" in st.session_state and st.session_state.last[0].strip():
    text, compare = st.session_state.last
    lab, sc, kind = predict(key, [text])
    pred = LABELS[int(lab[0])]
    c1, c2 = st.columns([1, 2])
    c1.markdown(f"### {pred}")
    c1.caption(model)
    if kind == "VADER compound":
        lo, hi = band(key)
        fig = go.Figure()
        fig.add_vrect(x0=-1, x1=lo, fillcolor=SENT["Negative"], opacity=0.25, line_width=0)
        fig.add_vrect(x0=lo, x1=hi, fillcolor=SENT["Neutral"], opacity=0.35, line_width=0)
        fig.add_vrect(x0=hi, x1=1, fillcolor=SENT["Positive"], opacity=0.25, line_width=0)
        fig.add_scatter(x=[sc[0]], y=[0], mode="markers", marker=dict(size=16, color="black"), showlegend=False)
        fig.update_xaxes(range=[-1, 1], title="VADER compound score"); fig.update_yaxes(visible=False)
        c2.plotly_chart(layout(fig, 160), width="stretch")
        c2.caption(f"Neutral band: {lo} to {hi}. Compound = {sc[0]:+.3f}")
    else:
        fig = go.Figure(go.Bar(x=sc[0] * 100, y=LABELS, orientation="h", marker_color=[SENT[l] for l in LABELS],
                               text=[f"{v * 100:.0f}%" for v in sc[0]], textposition="outside"))
        fig.update_xaxes(range=[0, 110], title=kind); fig.update_yaxes(autorange="reversed")
        c2.plotly_chart(layout(fig, 200), width="stretch")
    if compare:
        rows = []
        for name, k in MODELS.items():
            l2, s2, kd = predict(k, [text])
            conf = f"compound {s2[0]:+.2f}" if kd == "VADER compound" else f"{s2[0].max() * 100:.0f}%"
            rows.append({"Model": name, "Prediction": LABELS[int(l2[0])], "Score": conf,
                         "Study macro-F1 (CV)": "not evaluated in CV" if PERF.get(k) is None else f"{PERF[k]:.3f}"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if len({r["Prediction"] for r in rows}) > 1:
            st.caption("The models disagree. This is typical of mixed or lukewarm reviews, the cases the study found hardest.")
    if key.startswith("TFIDF"):
        with st.expander("Preprocessed text seen by the TF-IDF model"):
            st.code(clean_text(text) or "(empty after preprocessing)")

st.divider()
st.subheader("Classify a file")
st.caption("Upload a CSV with a column named `text`. Results stay in your browser session and are not stored.")
up = st.file_uploader("CSV file", type="csv")
if up is not None:
    df = pd.read_csv(up)
    if "text" not in df.columns:
        st.error("The file needs a column named `text`.")
    else:
        df = df.head(5000)
        lab, _, _ = predict(key, df.text.fillna("").astype(str).tolist())
        df["predicted_sentiment"] = [LABELS[int(i)] for i in lab]
        st.dataframe(df, width="stretch", height=300)
        share = df.predicted_sentiment.value_counts(normalize=True).reindex(LABELS).fillna(0) * 100
        st.markdown(" · ".join(f"**{l}** {share[l]:.1f}%" for l in LABELS))
        st.download_button("Download predictions", df.to_csv(index=False).encode(), "predictions.csv", "text/csv")
