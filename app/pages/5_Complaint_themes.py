import os, sys  # noqa: E401
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import TOPIC_NAMES, cat, csv, js, layout, setup, source

setup("Complaint themes", "💬")
st.title("Complaint themes (RQ4)")
st.markdown("Topics were extracted from the 319 Negative and Neutral reviews with non-negative matrix factorisation. "
            "Strongly evaluative words (e.g. *bad*, *terrible*) were removed first, so the topics describe **what** "
            "visitors complain about rather than how strongly.")

T = js("tourism", "tourism_results.json")["topic_model"]
rows = [{"Topic": TOPIC_NAMES[str(t["topic"])], "Reviews": t["n_reviews"], "Share (%)": round(t["share_pct"], 1),
         "% Negative within topic": round(t["pct_negative_within"], 1), "Top terms": ", ".join(t["top_terms"][:10]),
         "Categories (n)": "; ".join(f"{cat(k)} ({v})" for k, v in t["top_categories"].items())} for t in T["topics"]]
st.dataframe(pd.DataFrame(rows).sort_values("Reviews", ascending=False), hide_index=True, width="stretch")

st.subheader("Which themes dominate each category")
weight = st.radio("Shares", ["Rating-distribution-adjusted (post-stratified)", "Unweighted sample"], horizontal=True)
if weight.startswith("Rating"):
    M = csv("population", "topic_prevalence_reweighted_by_category.csv").set_index("Attraction_Category").fillna(0)
else:
    M = csv("tourism", "topic_by_category_pct.csv").set_index("Attraction_Category")
    M.columns = [TOPIC_NAMES[c] for c in M.columns.astype(str)]
order = ["Service encounter & staff", "Food quality & value for money", "Beach conditions & crowding",
         "Public-space setting (heterogeneous)"]
M = M[order]
fig = go.Figure(go.Heatmap(z=M.values, x=[o.replace(" & ", " &<br>").replace(" (", "<br>(") for o in order],
                           y=[cat(i) for i in M.index], colorscale="Blues", zmin=0, zmax=100, texttemplate="%{z:.0f}",
                           hovertemplate="%{y}: %{z:.1f}%<extra></extra>", colorbar=dict(title="%")))
fig.update_yaxes(autorange="reversed")
st.plotly_chart(layout(fig, 420), width="stretch")
O = csv("population", "topic_prevalence_overall.csv").set_index("Topic").loc[order]
st.markdown(f"The service-encounter and food-quality-and-value themes together account for "
            f"**{O.sample_share_pct.iloc[:2].sum():.0f}%** of negative and neutral reviews in the sample and "
            f"**{O.reweighted_share_pct.iloc[:2].sum():.0f}%** after rating-distribution adjustment. Beach dissatisfaction "
            "is driven by litter and crowding rather than the natural setting, which reviewers often praise in the same text.")

with st.expander("Choosing the number of topics"):
    K = csv("tourism", "topic_coherence.csv")
    fig = go.Figure(go.Scatter(x=K.k, y=K.c_v, mode="lines+markers", line=dict(color="#2a78d6", width=3)))
    fig.add_scatter(x=[T["selected_k"]], y=[K.set_index("k").c_v[T["selected_k"]]], mode="markers",
                    marker=dict(size=18, symbol="circle-open", color="black", line=dict(width=2)), showlegend=False)
    fig.update_xaxes(title="Number of topics k", dtick=1); fig.update_yaxes(title="Mean c_v coherence")
    st.plotly_chart(layout(fig, 300, showlegend=False), width="stretch")
    st.caption("k = 4 had the highest coherence, but the curve is nearly flat for k = 3, 6 and 10, so the four themes "
               "are best read as a descriptive map of complaint areas rather than a definitive taxonomy.")
source("tourism/tourism_results.json", "tourism/topic_by_category_pct.csv",
       "population/topic_prevalence_reweighted_by_category.csv", "population/topic_prevalence_overall.csv")
