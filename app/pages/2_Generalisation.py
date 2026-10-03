import os, sys  # noqa: E401
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import MODEL_C, NAMES, cat, ent, csv, layout, setup, source

setup("Generalisation", "🧭")
st.title("Generalisation to unseen entities (RQ2)")
st.markdown("Random cross-validation lets reviews of the same entity appear in training and test folds. "
            "**Leave-one-entity-out (LOEO)** cross-validation holds out all reviews of one entity at a time, "
            "so the model is tested on an attraction, beach, restaurant, hotel or venue it never saw in training.")

FAMS = ["VADER", "TFIDF-LR", "TFIDF-SVM", "BiLSTM", "CNN", "DistilBERT"]
nm = lambda f: "VADER (standard)" if f == "VADER" else NAMES[f]  # noqa: E731
G = csv("generalisation", "cv_comparison.csv")
fig = go.Figure()
for sch, sym, lab, dy in [("random_5fold", "circle", "Random 5-fold CV", -0.15), ("leave_one_attraction_out", "square-open", "LOEO CV", 0.15)]:
    g = G[G.scheme == sch].set_index("family")
    for i, f in enumerate(FAMS):
        r = g.loc[f]
        fig.add_scatter(x=[r.macro_f1], y=[i + dy], mode="markers", name=lab, legendgroup=lab, showlegend=(i == 0),
                        marker=dict(symbol=sym, size=11, color=MODEL_C[f], line=dict(width=2, color=MODEL_C[f])),
                        error_x=dict(type="data", symmetric=False, array=[r.ci_hi - r.macro_f1],
                                     arrayminus=[r.macro_f1 - r.ci_lo], thickness=2, color=MODEL_C[f], width=4),
                        hovertemplate=f"{nm(f)}: %{{x:.3f}} [{r.ci_lo:.3f}, {r.ci_hi:.3f}]<extra>{lab}</extra>")
fig.update_yaxes(tickvals=list(range(len(FAMS))), ticktext=[nm(f) for f in FAMS], autorange="reversed")
fig.update_xaxes(title="Pooled macro-F1 over all 598 reviews (bootstrap 95% CI)")
st.plotly_chart(layout(fig, 400), width="stretch")

D = csv("generalisation", "random_vs_loao_drop.csv").set_index("family").loc[FAMS].reset_index()
st.dataframe(pd.DataFrame({
    "Model": D.family.map(nm), "Random 5-fold": D.random_macro_f1.round(3), "LOEO": D.loao_macro_f1.round(3),
    "Drop [95% CI]": ["—" if f == "VADER" else f"{d:+.3f} [{l:.3f}, {h:.3f}]" for f, d, l, h in
                      zip(D.family, D["drop"], D.drop_ci_lo, D.drop_ci_hi)],
    "McNemar p (unadjusted)": ["—" if f == "VADER" else f"{p:.3f}" for f, p in zip(D.family, D.mcnemar_p)]}),
    hide_index=True, width="stretch")
st.caption("DistilBERT shows no detectable generalisation penalty; every drop has a bootstrap interval that includes zero. "
           "VADER needs no training and scores the same under both schemes.")

st.subheader("Accuracy on each held-out entity")
A = csv("generalisation", "per_attraction_loao.csv")
metric = st.radio("Metric", ["LOEO accuracy", "LOEO macro-F1"], horizontal=True)
col = "loao_acc" if metric == "LOEO accuracy" else "loao_macro_f1"
A["Entity"] = A.attraction.map(ent) + " (" + A.category.map(cat) + ", n=" + A.n.astype(str) + ")"
piv = A.pivot_table(index="Entity", columns="family", values=col)[FAMS]
piv = piv.sort_values("DistilBERT")
fig = go.Figure(go.Heatmap(z=piv.values * 100, x=[nm(f) for f in FAMS], y=piv.index, colorscale="Blues", zmin=0, zmax=100,
                           texttemplate="%{z:.0f}", hovertemplate="%{y}<br>%{x}: %{z:.1f}%<extra></extra>",
                           colorbar=dict(title="%")))
st.plotly_chart(layout(fig, 560), width="stretch")
st.caption("Performance varies far more between entities than between schemes. The Clock Tower and Ilıca Plajı, both with "
           "many Neutral reviews, are hardest; small entities (n < 10) give unstable estimates.")
source("generalisation/cv_comparison.csv", "generalisation/random_vs_loao_drop.csv", "generalisation/per_attraction_loao.csv")
