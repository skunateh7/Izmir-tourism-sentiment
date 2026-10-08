import os, sys  # noqa: E401
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import SENT, cat, ent, csv, js, layout, setup, source, stacked_sentiment, wilson

setup("Destination sentiment", "🗺️")
st.title("Destination-level ratings and sentiment (RQ3)")
st.markdown("Reviews were collected with TripAdvisor's rating filters so that all three classes were represented, "
            "which over-represents negative reviews. Destination-level sentiment is therefore computed from each "
            "entity's **full TripAdvisor rating histogram** (all languages), mapped to sentiment classes with the same rule. "
            "These shares describe **ratings**; the expander below gives an audit-based estimate for review text.")

P = js("population", "population_results.json")
pop, smp = P["population_overall_pct"], P["sample_overall_pct"]
c = st.columns(4)
c[0].metric("All ratings", f"{P['N_population_reviews']:,}")
c[1].metric("Positive", f"{pop['Positive']:.1f}%")
c[2].metric("Negative", f"{pop['Negative']:.1f}%",
            delta=f"{smp['Negative']:.1f}% in the collected sample", delta_color="off")
c[3].metric("Negative over-representation", f"{smp['Negative'] / pop['Negative']:.1f}×")

with st.expander("Estimated sentiment of review text (audit-based)"):
    TE = csv("review2", "text_based_population_estimates.csv")
    TE = TE[(TE.level != "Attraction")].copy()
    TE["Unit"] = [("All entities" if l == "Overall" else cat(u)) for l, u in zip(TE.level, TE.unit)]
    TE["Audit labels"] = TE.label_source.map({"A": "Annotator A", "B": "Annotator B (author)", "agreed": "Agreed only"})
    for k in ["Positive", "Neutral", "Negative"]:
        TE[f"Text {k} % [95% CI]"] = [f"{v:.1f} [{a:.1f}, {b:.1f}]" for v, a, b in zip(TE[f"text_{k}"], TE[f"text_{k}_lo"], TE[f"text_{k}_hi"])]
    TE["Ratings Pos / Neu / Neg %"] = [f"{a:.1f} / {b:.1f} / {c:.1f}" for a, b, c in zip(TE.rating_Positive, TE.rating_Neutral, TE.rating_Negative)]
    st.dataframe(TE[["Unit", "Audit labels", "Ratings Pos / Neu / Neg %", "Text Positive % [95% CI]", "Text Neutral % [95% CI]",
                     "Text Negative % [95% CI]"]], hide_index=True, width="stretch")
    st.caption("The label audit's rates of positive, neutral and negative text within each rating group are applied to each "
               "histogram (bootstrap over the 141 audited reviews). The positive share barely changes; the negative share "
               "rises because most 3-star texts read as negative or positive. Assumes the rating–text relation is the same "
               "across entities.")

st.header("By category")
C = csv("population", "population_sentiment_by_category.csv").sort_values("net_pop", ascending=False)
C["label"] = [f"{cat(c)}  (N={n:,})" for c, n in zip(C.Category, C.N_pop)]
st.plotly_chart(stacked_sentiment(C.rename(columns={"pop_neg": "neg", "pop_neu": "neu", "pop_pos": "pos"}), "label", h=330,
                                  xtitle="Share of all TripAdvisor ratings (%)"), width="stretch")
X = P["category_x_sentiment_population"]
st.markdown(f"Category and sentiment are associated, but modestly: χ²({X['dof']}) = {X['chi2']:.1f}, "
            f"p < 0.001, Cramér's V = {X['cramers_v']:.2f}.")
res = pd.DataFrame(X["std_residuals"]).T
res.index = [cat(i) for i in res.index]
with st.expander("Standardised residuals (positive = more ratings than expected under independence)"):
    st.dataframe(res[["Negative", "Neutral", "Positive"]].round(1),
                 width="stretch")

st.header("By entity")
A = csv("population", "sample_vs_population.csv")
Pa = csv("population", "population_sentiment_by_attraction.csv").set_index("Attraction_Name")
ci = [wilson(int(Pa.loc[a, "s1"] + Pa.loc[a, "s2"]), int(Pa.loc[a, "N_pop"])) for a in A.Attraction_Name]
A["lo"], A["hi"] = [x[0] for x in ci], [x[1] for x in ci]
cats = ["All"] + sorted({cat(c) for c in A.Category})
pick = st.selectbox("Category", cats)
if pick != "All":
    A = A[A.Category.map(cat) == pick]
A = A.sort_values("pop_neg", ascending=False)
A["label"] = [f"{ent(a)}  [{cat(c)}; N={n:,}]" for a, c, n in zip(A.Attraction_Name, A.Category, A.N_pop)]
fig = go.Figure()
fig.add_scatter(x=A.pop_neg, y=A.label, mode="markers", name="All TripAdvisor ratings (Wilson 95% CI)",
                marker=dict(size=11, color=SENT["Negative"]),
                error_x=dict(type="data", symmetric=False, array=A.hi - A.pop_neg, arrayminus=A.pop_neg - A.lo,
                             color=SENT["Negative"], thickness=2))
fig.add_scatter(x=A.sample_neg, y=A.label, mode="markers", name="Collected sample",
                marker=dict(size=10, symbol="circle-open", color=SENT["Negative"], line=dict(width=2)))
fig.update_xaxes(title="Negative ratings, 1–2 stars (%)", range=[-2, 75], zeroline=False)
fig.update_yaxes(autorange="reversed")
st.plotly_chart(layout(fig, 120 + 34 * len(A)), width="stretch")
st.caption(f"The ranking of entities is broadly preserved between sample and all ratings "
           f"(Spearman ρ = {P['rank_correlation_net_sentiment_sample_vs_population']:.2f} for net sentiment), "
           "but the sample's negative shares are far too high, and this agreement concerns ratings only, not how often each "
           "complaint theme occurs. Small venues (e.g. Bios Bar, N = 24) have wide intervals.")
show = A[["Attraction_Name", "Category", "N_pop", "pop_pos", "pop_neu", "pop_neg", "n_sample", "sample_neg"]].copy()
show["Category"] = show.Category.map(cat)
show["Attraction_Name"] = show.Attraction_Name.map(ent)
for c in ["pop_pos", "pop_neu", "pop_neg", "sample_neg"]:
    show[c] = show[c].map(lambda v: f"{v:.1f}")
show["N_pop"] = show.N_pop.map(lambda v: f"{v:,}")
show.columns = ["Entity", "Category", "All ratings", "Positive %", "Neutral %", "Negative %", "Sample n", "Sample negative %"]
st.dataframe(show, hide_index=True, width="stretch")
source("population/population_results.json", "population/population_sentiment_by_category.csv",
       "population/sample_vs_population.csv")
