"""İzmir tourist-review sentiment - companion app to the manuscript.

Run from the repository root:   streamlit run app/Home.py
"""
import pandas as pd
import streamlit as st

from lib import TITLE, NAMES, csv, js, setup, source

setup("Home", "🏛️")
st.title("Tourist-review sentiment for destination intelligence: İzmir")
st.markdown(f"Companion app to the manuscript *{TITLE}* (under review).")

G = csv("generalisation", "cv_comparison.csv")
dist_r = G[(G.family == "DistilBERT") & (G.scheme == "random_5fold")].iloc[0]
dist_l = G[(G.family == "DistilBERT") & (G.scheme == "leave_one_attraction_out")].iloc[0]
lr_r = G[(G.family == "TFIDF-LR") & (G.scheme == "random_5fold")].iloc[0]
P = js("population", "population_results.json")

k = st.columns(5)
k[0].metric("Audited reviews", "598", help="TripAdvisor reviews of 14 tourism entities in six categories, collected September 2026")
k[1].metric("All TripAdvisor ratings", f"{P['N_population_reviews']:,}", help="Full rating histograms of the 14 entities, used to adjust for rating-filtered sampling")
k[2].metric("DistilBERT macro-F1", f"{dist_r.macro_f1:.3f}", help="Pooled random 5-fold cross-validation, N = 598")
k[3].metric("vs best TF-IDF", f"+{dist_r.macro_f1 - lr_r.macro_f1:.3f}", help="TF-IDF + LR, pooled random 5-fold CV")
k[4].metric("Unseen entities (LOEO)", f"{dist_l.macro_f1:.3f}", help="Leave-one-entity-out cross-validation")

st.page_link("pages/6_Analyse_a_review.py", label="Analyse your own review with the study's models →", icon="🎯")

st.subheader("What the study found")
st.markdown(f"""
1. **Pretraining, not architectural depth, made the difference (RQ1).** A fine-tuned DistilBERT classified sentiment
   best (pooled cross-validated macro-F1 {dist_r.macro_f1:.2f}); CNN and BiLSTM models trained from scratch were no
   better than tuned TF-IDF classifiers.
2. **Models transfer to unseen tourism entities (RQ2).** DistilBERT showed no detectable penalty under
   leave-one-entity-out cross-validation ({dist_r.macro_f1:.3f} random vs {dist_l.macro_f1:.3f} LOEO).
3. **The Neutral class is a measurement problem as much as a modelling one.** A blind text-only audit confirmed nearly
   all star-derived positive and negative labels but only a minority of 3-star "neutral" labels, many of which are
   mixed experiences.
4. **Visitor sentiment is predominantly positive (RQ3).** Across all {P['N_population_reviews']:,} ratings,
   {P['population_overall_pct']['Positive']:.1f}% are positive and {P['population_overall_pct']['Negative']:.1f}% negative;
   gastronomy and nightlife concentrate negative experiences.
5. **Complaints centre on the service encounter and value for money (RQ4),** and at the beach on litter and crowding.
""")

st.subheader("Explore")
st.markdown("""
| Page | What it shows |
|---|---|
| **Model comparison** | Test-set and cross-validated scores, confusion matrices and paired tests (RQ1) |
| **Generalisation** | Random vs leave-one-entity-out cross-validation, per-entity accuracy (RQ2) |
| **Neutral class & labels** | Label audit, error analysis and ablation |
| **Destination sentiment** | Category and entity sentiment from all TripAdvisor ratings (RQ3) |
| **Complaint themes** | Topic model of negative and neutral reviews (RQ4) |
| **Analyse a review** | Type your own review and see how the study's models classify it |
| **About** | Methods summary, reproducibility, data and AI-use statements |
""")
st.info("All numbers in this app are read directly from the result files produced by the study's scripts, "
        "so they match the manuscript. Review texts are not redistributed.")
source("generalisation/cv_comparison.csv", "population/population_results.json")
