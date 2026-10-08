"""Shared loaders, labels and plotting helpers for the Streamlit app.

Every number shown in the app is read from the study's results/ folder, so the app always agrees
with the manuscript. Nothing here contains or loads review texts.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(APP)
RES = os.path.join(ROOT, "results")
sys.path.insert(0, ROOT)

TITLE = ("From Tourist Reviews to Destination Intelligence: Pretraining, "
         "Cross-Entity Generalisation and Rating–Text Disagreement in İzmir")
LABELS = ["Negative", "Neutral", "Positive"]
SENT = {"Negative": "#e34948", "Neutral": "#b9b8b2", "Positive": "#2a78d6"}
MODEL_C = {"VADER-std": "#f2a07a", "VADER": "#eb6834", "TFIDF-LR": "#1baf7a", "TFIDF-SVM": "#008300",
           "BiLSTM": "#4a3aa7", "CNN": "#2a78d6", "DistilBERT": "#e87ba4"}
NAMES = {"VADER-std": "VADER (standard)", "VADER": "VADER (tuned band)", "TFIDF-LR": "TF-IDF + LR",
         "TFIDF-SVM": "TF-IDF + SVM", "BiLSTM": "BiLSTM", "CNN": "Multi-kernel CNN", "DistilBERT": "DistilBERT"}
ORDER = ["VADER-std", "VADER", "TFIDF-LR", "TFIDF-SVM", "BiLSTM", "CNN", "DistilBERT"]
CAT = {"Family": "Accommodation"}  # data label -> manuscript label
TOPIC_NAMES = {"0": "Service encounter & staff", "1": "Beach conditions & crowding",
               "2": "Public-space setting (heterogeneous)", "3": "Food quality & value for money"}


ENT = {"Bostanli Meyhanesi": "Bostanlı Meyhanesi", "Ilica Plaji": "Ilıca Plajı", "Izmir Marriott Hotel": "İzmir Marriott Hotel",
       "Izmir Saat Kulesi (Clock Tower)": "İzmir Clock Tower", "Izmir Wildlife Park": "İzmir Wildlife Park",
       "Tavaci Recep Usta": "Tavaçı Recep Usta"}  # data spelling -> manuscript spelling


def cat(c):
    return CAT.get(c, c)


def ent(e):
    return ENT.get(e, e)


def R(*p):
    return os.path.join(RES, *p)


@st.cache_data
def csv(*p):
    return pd.read_csv(R(*p))


@st.cache_data
def js(*p):
    with open(R(*p), encoding="utf-8") as f:
        return json.load(f)


def setup(title, icon="📊"):
    st.set_page_config(page_title=f"{title} · İzmir review sentiment", page_icon=icon, layout="wide")
    with st.sidebar:
        st.markdown("**İzmir tourist-review sentiment**")
        st.caption("Companion app to the manuscript. All figures are read from the study's result files.")


def layout(fig, h=380, **kw):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(size=13),
                      legend=dict(orientation="h", y=1.08, x=0), **kw)
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)")
    return fig


def stacked_sentiment(df, ycol, cols=("neg", "neu", "pos"), h=380, xtitle="Share (%)"):
    fig = go.Figure()
    for lab, c in zip(LABELS, cols):
        fig.add_bar(y=df[ycol], x=df[c], name=lab, orientation="h", marker_color=SENT[lab],
                    text=[f"{v:.0f}%" if v >= 6 else "" for v in df[c]], textposition="inside",
                    hovertemplate=f"{lab}: %{{x:.1f}}%<extra></extra>")
    fig.update_layout(barmode="stack", legend_traceorder="normal")
    fig.update_xaxes(range=[0, 100], title=xtitle)
    fig.update_yaxes(autorange="reversed")
    return layout(fig, h)


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - hw) * 100, (c + hw) * 100


@st.cache_data
def main_table():
    """Table 3 of the manuscript (test set, n = 90). Full-precision values from results/final; the standard-VADER
    row (not part of model selection) comes from the paper table."""
    f = csv("final", "table_main_results.csv")
    if "VADER-std" not in set(f.Model):
        p = csv("paper", "tables", "table2_main_results.csv")
        f = pd.concat([p[p.Model == "VADER-std"], f], ignore_index=True)
    return f


def source(*files):
    st.caption("Source: " + ", ".join(f"`results/{f}`" for f in files))
