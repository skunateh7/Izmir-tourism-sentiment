import os, sys  # noqa: E401
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import MODEL_C, NAMES, cat, ent, csv, js, layout, setup, source

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
st.caption("With configurations fixed in advance, every drop has a bootstrap interval that includes zero. A non-significant "
           "drop does not by itself show no loss, so the expander below tests equivalence against a 0.05 margin. VADER needs no "
           "training and scores the same under both schemes.")

with st.expander("Equivalence test: is the loss on unseen entities smaller than 0.05 macro-F1?"):
    EQ = csv("review2", "equivalence_table.csv")
    import ast
    c2 = lambda v: "[{:.3f}, {:.3f}]".format(*ast.literal_eval(v))
    st.dataframe(pd.DataFrame({"Model": EQ.model.map(nm), "Protocol": EQ.protocol.map({"fixed": "Validation-selected (primary)", "nested": "Nested selection"}),
                               "Drop": EQ["drop"].round(3), "90% CI (reviews)": EQ.ci90_review.map(c2),
                               "90% CI (entities)": EQ.ci90_entity.map(c2),
                               "Equivalent within ±0.05": [("yes" if a and b else "reviews only" if a else "entities only" if b else "no")
                                                           for a, b in zip(EQ.tost_equivalent_review, EQ.tost_equivalent_entity)]}),
                 hide_index=True, width="stretch")
    st.caption("Two one-sided tests at α = 0.05: equivalent if the 90% interval of the drop lies within ±0.05. For DistilBERT "
               "a loss above about 0.04 can be excluded under both protocols; smaller losses cannot. The margin was set after "
               "review as roughly the loss shown by tuned TF-IDF.")

with st.expander("Sensitivity: TF-IDF with configuration selection nested within each fold"):
    N = js("generalisation", "nested_tfidf.json")
    rows = []
    for f in N:
        for sch, lab in [("random_5fold", "Random 5-fold"), ("leave_one_attraction_out", "LOEO")]:
            r = N[f][sch]
            rows.append({"Model": NAMES[f], "Scheme": lab, "Fixed configuration": f"{r['non_nested_macro_f1']:.3f}",
                         "Nested selection [95% CI]": f"{r['macro_f1']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]",
                         "DistilBERT minus nested [95% CI]": f"{r['distilbert_minus_nested']['value']:+.3f} "
                         f"[{r['distilbert_minus_nested']['ci'][0]:.3f}, {r['distilbert_minus_nested']['ci'][1]:.3f}]"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.markdown(" · ".join(f"**{NAMES[f]}** nested drop {N[f]['drop']['value']:.3f} "
                           f"[{N[f]['drop']['ci'][0]:.3f}, {N[f]['drop']['ci'][1]:.3f}]" for f in N))
    st.caption("When TF-IDF is tuned using only each fold's training data, it scores higher within known entities and shows a "
               "modest but detectable drop on unseen entities, while DistilBERT's advantage remains. Nesting was not feasible "
               "for the CNN and BiLSTM. All results concern entities within one destination.")

with st.expander("Sensitivity: DistilBERT with each of its four configurations, and nested selection"):
    D = js("generalisation", "distilbert_configs_cv.json")
    CN = {"lr=2e-05,cw=none": "lr 2e-5, unweighted", "lr=2e-05,cw=bal": "lr 2e-5, class-weighted (selected)",
          "lr=5e-05,cw=none": "lr 5e-5, unweighted", "lr=5e-05,cw=bal": "lr 5e-5, class-weighted",
          "nested selection": "Nested selection in each fold"}
    f3 = lambda v, c: f"{v:.3f} [{c[0]:.3f}, {c[1]:.3f}]"
    st.dataframe(pd.DataFrame([{"Configuration": CN[r["config"]],
                                "Random 5-fold [95% CI]": f3(r["random_5fold_macro_f1"], r["random_5fold_ci"]),
                                "LOEO [95% CI]": f3(r["leave_one_attraction_out_macro_f1"], r["leave_one_attraction_out_ci"]),
                                "Drop [95% CI]": f3(r["drop_random_minus_loeo"]["diff"], r["drop_random_minus_loeo"]["diff_ci95"])}
                               for r in D["configs"]]), hide_index=True, width="stretch")
    st.caption("All four DistilBERT configurations were run with the same folds and seeds. Nested selection picks, in each "
               "outer fold, the configuration with the best score on an inner validation split drawn only from the training "
               "entities. Under nested selection DistilBERT's drop is small (equivalent within 0.05); only the "
               "higher-learning-rate, class-weighted configuration shows a small detectable drop.")

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
