import os, sys  # noqa: E401
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import LABELS, MODEL_C, NAMES, ORDER, csv, js, layout, main_table, setup, source

f3 = "{:.3f}".format
setup("Model comparison", "🧪")
st.title("Model comparison (RQ1)")
st.markdown("Six approaches were compared under validation-only model selection: a lexicon baseline (VADER), "
            "TF-IDF with logistic regression (LR) or a linear SVM, a multi-kernel CNN and a BiLSTM trained from "
            "scratch, and a fine-tuned DistilBERT transformer.")

view = st.radio("Evaluation", ["Pooled cross-validation (N = 598)", "Locked test set (n = 90)"], horizontal=True)

if view.startswith("Locked"):
    T = main_table().set_index("Model").loc[ORDER].reset_index()
    neural = T.n_runs > 1
    fig = go.Figure()
    for _, r in T.iterrows():
        fig.add_scatter(x=[r.Rep_MacroF1], y=[NAMES[r.Model]], mode="markers", marker=dict(size=12, color=MODEL_C[r.Model]),
                        error_x=dict(type="data", symmetric=False, array=[r.Rep_MacroF1_CI_hi - r.Rep_MacroF1],
                                     arrayminus=[r.Rep_MacroF1 - r.Rep_MacroF1_CI_lo], thickness=2, color=MODEL_C[r.Model]),
                        showlegend=False, hovertemplate="%{y}: %{x:.3f}<extra></extra>")
    fig.update_xaxes(title="Test macro-F1 with bootstrap 95% CI (deterministic output or five-seed ensemble)", range=[0.3, 0.95])
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(layout(fig, 360), width="stretch")
    show = pd.DataFrame({
        "Model": T.Model.map(NAMES),
        "Accuracy": [f"{a:.3f} ± {s:.3f}" if n else f"{a:.3f}" for a, s, n in zip(T.Accuracy, T.Accuracy_SD, neural)],
        "Macro-F1": [f"{a:.3f} ± {s:.3f}" if n else f"{a:.3f}" for a, s, n in zip(T.Macro_F1, T.Macro_F1_SD, neural)],
        "95% CI": [f"{lo:.2f}–{hi:.2f}" for lo, hi in zip(T.Rep_MacroF1_CI_lo, T.Rep_MacroF1_CI_hi)],
        "F1 Neg": T.F1_Negative.map(f3), "F1 Neu": T.F1_Neutral.map(f3), "F1 Pos": T.F1_Positive.map(f3)})
    st.dataframe(show, hide_index=True, width="stretch")
    st.caption("Manuscript Table 3. Neural models: accuracy, macro-F1 and per-class F1 are means over five seeds "
               "(± SD for accuracy and macro-F1); the CI refers to the five-seed ensemble.")

    st.subheader("Confusion matrices")
    F = js("final", "final_results.json")["details"]
    fam = st.selectbox("Model", [m for m in ORDER if m in F], index=len([m for m in ORDER if m in F]) - 1,
                       format_func=lambda m: NAMES[m])
    cm = np.array(F[fam]["metrics"]["confusion_matrix"])
    pct = cm / cm.sum(1, keepdims=True) * 100
    fig = go.Figure(go.Heatmap(z=pct, x=LABELS, y=LABELS, colorscale="Blues", zmin=0, zmax=100, showscale=False,
                               text=[[f"{cm[i, j]}<br>{pct[i, j]:.0f}%" for j in range(3)] for i in range(3)],
                               texttemplate="%{text}", hoverinfo="skip"))
    fig.update_xaxes(title="Predicted"); fig.update_yaxes(title="True", autorange="reversed")
    c1, c2 = st.columns([1, 1])
    c1.plotly_chart(layout(fig, 340), width="stretch")
    c2.markdown("Positive reviews are recognised best and Neutral reviews worst by every model. DistilBERT improves "
                "mainly on Negative and Neutral reviews and makes almost no Positive–Negative confusions.")
    st.subheader("Paired comparisons on the test set")
    P = csv("final", "table_pairwise_tests.csv")
    SRC = ("final/table_main_results.csv", "final/final_results.json", "final/table_pairwise_tests.csv")
else:
    G = csv("generalisation", "cv_comparison.csv")
    G = G[G.scheme == "random_5fold"].set_index("family").loc[["VADER", "TFIDF-LR", "TFIDF-SVM", "BiLSTM", "CNN", "DistilBERT"]].reset_index()
    nm = lambda f: "VADER (standard)" if f == "VADER" else NAMES[f]  # noqa: E731
    fig = go.Figure()
    for _, r in G.iterrows():
        fig.add_scatter(x=[r.macro_f1], y=[nm(r.family)], mode="markers", marker=dict(size=12, color=MODEL_C[r.family]),
                        error_x=dict(type="data", symmetric=False, array=[r.ci_hi - r.macro_f1],
                                     arrayminus=[r.macro_f1 - r.ci_lo], thickness=2, color=MODEL_C[r.family]),
                        showlegend=False, hovertemplate="%{y}: %{x:.3f}<extra></extra>")
    fig.update_xaxes(title="Pooled out-of-fold macro-F1, random 5-fold CV (bootstrap 95% CI)", range=[0.4, 0.8])
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(layout(fig, 340), width="stretch")
    st.dataframe(pd.DataFrame({"Model": G.family.map(nm), "Macro-F1": G.macro_f1.map(f3),
                               "95% CI": [f"{a:.3f}–{b:.3f}" for a, b in zip(G.ci_lo, G.ci_hi)],
                               "Accuracy": G.accuracy.map(f3), "F1 Neg": G.f1_Negative.map(f3),
                               "F1 Neu": G.f1_Neutral.map(f3), "F1 Pos": G.f1_Positive.map(f3)}),
                 hide_index=True, width="stretch")
    st.subheader("Paired comparisons (pooled CV)")
    P = csv("generalisation", "pairwise_tests_cv.csv").rename(columns={"macroF1_diff": "macroF1_diff_A_minus_B"})
    SRC = ("generalisation/cv_comparison.csv", "generalisation/pairwise_tests_cv.csv")

nm2 = lambda f: NAMES.get(f, f)  # noqa: E731
P = P.assign(**{"Model A": P.A.map(nm2), "Model B": P.B.map(nm2),
                "Macro-F1 diff (A−B) [95% CI]": [f"{a:+.3f} [{l:.3f}, {h:.3f}]" for a, l, h in
                                                  zip(P.macroF1_diff_A_minus_B, P.diff_ci_lo, P.diff_ci_hi)],
                "McNemar p (Holm)": P.mcnemar_p_holm.map(lambda v: f"{v:.3f}")})
only_dist = st.checkbox("Show only comparisons with DistilBERT", value=True)
if only_dist:
    P = P[(P.A == "DistilBERT") | (P.B == "DistilBERT")]
st.dataframe(P[["Model A", "Model B", "Macro-F1 diff (A−B) [95% CI]", "McNemar p (Holm)"]], hide_index=True,
             width="stretch")
st.caption("Bootstrap intervals refer to the macro-F1 difference; the McNemar test compares per-review correctness "
           "(Holm-adjusted across all 15 pairs).")

with st.expander("Sensitivity: excluding the 90 reviews used for model selection"):
    S = csv("generalisation", "cv_excluding_selection_reviews.csv")
    S = S[S.scheme == "random_5fold"][["family", "all598", "excl_val508"]]
    S[["all598", "excl_val508"]] = S[["all598", "excl_val508"]].apply(lambda c: c.map(f3))
    S.columns = ["Model", "All 598 reviews", "508 reviews never used for selection"]
    S["Model"] = S.Model.map(lambda f: "VADER (standard)" if f == "VADER" else NAMES[f])
    st.dataframe(S, hide_index=True, width="stretch")
    D = csv("generalisation", "distilbert_vs_others_excl_selection.csv")
    D = D.assign(Scheme=D.scheme.map({"random_5fold": "Random 5-fold", "leave_one_attraction_out": "Leave-one-entity-out"}),
                 vs=D.vs.map(lambda f: "VADER (standard)" if f == "VADER" else NAMES[f]),
                 diff=[f"{a:+.3f} [{l:.3f}, {h:.3f}]" for a, l, h in zip(D["diff"], D.ci_lo, D.ci_hi)])
    st.dataframe(D[["Scheme", "vs", "diff"]].rename(columns={"vs": "DistilBERT vs", "diff": "Macro-F1 difference [95% CI]"}),
                 hide_index=True, width="stretch")
    st.caption("Conclusions are unchanged when scored only on the reviews never used to choose configurations.")
source(*SRC, "generalisation/cv_excluding_selection_reviews.csv")
