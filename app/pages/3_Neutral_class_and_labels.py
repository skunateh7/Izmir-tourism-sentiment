import os, sys  # noqa: E401
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import MODEL_C, NAMES, SENT, cat, csv, js, layout, setup, source

setup("Neutral class & labels", "🔍")
st.title("Why the Neutral class is hard")
st.markdown("Labels were derived from star ratings (1–2 = Negative, 3 = Neutral, 4–5 = Positive). Every model found "
            "Neutral reviews hardest. A blind label audit and an error analysis show that this is largely a "
            "**measurement** problem: many 3-star reviews do not read as neutral when judged from the text alone.")

st.header("Blind label audit")
L = js("label_audit", "label_agreement_main_vs_overrides.json")
M = L["random_stratified_non_override"]
c = st.columns(4)
c[0].metric("Audited reviews", M["n"], help="Randomly drawn within each rating-derived class; the eight manual overrides are evaluated separately")
c[1].metric("Annotator agreement", f"{M['A_vs_B']['pct']:.1f}%")
c[2].metric("Cohen's κ", f"{M['A_vs_B']['kappa']:.2f}", help=f"Bootstrap 95% CI {M['A_vs_B']['ci'][0]:.2f}–{M['A_vs_B']['ci'][1]:.2f}")
c[3].metric("Consensus reviews", M["consensus_n"], help="Reviews on which both annotators agreed")

cons = M["consensus_vs_study"]
rows = []
for lab in ["Positive", "Negative", "Neutral"]:
    k, n = map(int, cons[lab].split("/"))
    rows.append((lab, k, n))
fig = go.Figure(go.Bar(x=[k / n * 100 for _, k, n in rows], y=[f"Star-derived {l} (n={n})" for l, _, n in rows],
                       orientation="h", marker_color=[SENT[l] for l, _, _ in rows],
                       text=[f"{k}/{n}" for _, k, n in rows], textposition="outside"))
fig.update_xaxes(title="Confirmed by both text-only annotators (%)", range=[0, 110])
fig.update_yaxes(autorange="reversed")
c1, c2 = st.columns([3, 2])
c1.plotly_chart(layout(fig, 260), width="stretch")
nb = M["consensus_neutral_breakdown"]
c2.markdown(f"Of the **{sum(nb.values())}** star-derived Neutral reviews on which the annotators agreed, they read "
            f"**{nb['Neutral']}** as neutral, **{nb['Negative']}** as negative and **{nb['Positive']}** as positive.")
c2.caption("Text-only annotators disagreeing with a star-derived label is evidence of rating–text mismatch, not proof "
           "that the label is wrong; there is no absolute ground truth.")

st.header("Where the best model still fails")
E = csv("errors", "error_rates_by_factor.csv")
NICE = {("three_star", "True"): "3-star review", ("three_star", "False"): "1, 2, 4 or 5 stars",
        ("contrast_marker", "True"): "Contrast marker present", ("contrast_marker", "False"): "No contrast marker",
        ("explicit_negation", "True"): "Explicit negation", ("explicit_negation", "False"): "No negation",
        ("length_bin", "<=25"): "≤ 25 words", ("length_bin", "26-45"): "26–45 words",
        ("length_bin", "46-75"): "46–75 words", ("length_bin", ">75"): "> 75 words"}
E["group"] = [NICE.get((f, str(l))) or (cat(l) if f == "category" else None) for f, l in zip(E.factor, E.level)]
E = E[E.group.notna() & E.model.isin(["DistilBERT", "TFIDF-LR"])]
fig = go.Figure()
for m in ["TFIDF-LR", "DistilBERT"]:
    e = E[E.model == m]
    fig.add_bar(y=e.group, x=e.error_rate * 100, name=NAMES[m], orientation="h", marker_color=MODEL_C[m],
                customdata=e.n, hovertemplate="%{y} (n=%{customdata}): %{x:.1f}%<extra>" + NAMES[m] + "</extra>")
fig.update_layout(barmode="group"); fig.update_yaxes(autorange="reversed")
fig.update_xaxes(title="Error rate (%), pooled random 5-fold CV, N = 598")
st.plotly_chart(layout(fig, 620), width="stretch")
st.caption("Three-star reviews are misclassified three times as often as others, and contrast markers roughly double "
           "the error rate, even for DistilBERT.")

st.subheader("Qualitative error coding (50 DistilBERT errors, two coders)")
C = csv("errors", "second_coder_codes.csv")
A2 = js("errors", "second_coder_agreement.json")["primary_agreement"]
LAB = {"mixed_sentiment": "Mixed sentiment", "rating_text_mismatch": "Rating–text mismatch",
       "negation_or_contrast": "Negation or contrast", "neutral_boundary": "Weak polarity",
       "non_standard_language": "Non-standard language", "implicit_sentiment": "Implicit sentiment",
       "sarcasm_irony": "Sarcasm or irony", "short_or_uninformative": "Short or uninformative"}
cats = list(LAB)
fig = go.Figure()
for col, name, color in [("Final_primary", "Author", "#e87ba4"), ("c2", "Second coder (blind)", "#4a3aa7")]:
    v = C[col].value_counts().reindex(cats).fillna(0)
    fig.add_bar(y=[LAB[c] for c in cats], x=v.values, name=name, orientation="h", marker_color=color,
                text=v.astype(int).values, textposition="outside")
fig.update_layout(barmode="group"); fig.update_yaxes(autorange="reversed")
fig.update_xaxes(title="Primary cause (number of errors)")
st.plotly_chart(layout(fig, 420), width="stretch")
st.caption(f"Agreement on the primary cause was only fair: {A2['agree']} of {A2['n']} ({A2['pct']:.0f}%), Cohen's κ = "
           f"{A2['kappa']:.2f} (95% CI {A2['ci'][0]:.2f}–{A2['ci'][1]:.2f}). The categories overlap (mixed sentiment vs "
           "contrast), so the counts are exploratory. The author's codes started from AI-proposed provisional categories; "
           "the second coder worked blind and without AI tools.")

with st.expander("Ablation (Supplementary Table S3)"):
    A = csv("ablation", "ablation.csv")
    st.dataframe(pd.DataFrame({
        "Model": A.model.map(lambda m: NAMES.get(m, m)), "Variant": A.variant.str.replace(" (v5)", "", regex=False).str.capitalize(),
        "Validation macro-F1": [f"{m:.3f}" + (f" ± {s:.3f}" if s > 0 else "") for m, s in zip(A.val_mean, A.val_sd)],
        "Test macro-F1": [f"{m:.3f}" + (f" ± {s:.3f}" if s > 0 else "") for m, s in zip(A.test_mean, A.test_sd)],
        "Test F1 Neutral": A.test_f1_Neutral.map("{:.3f}".format)}), hide_index=True, width="stretch",
        height=35 * (len(A) + 1) + 3)
source("label_audit/label_agreement_main_vs_overrides.json", "errors/error_rates_by_factor.csv",
       "errors/second_coder_codes.csv", "ablation/ablation.csv")
