"""
Step 12 — Correct the rating-filtered sample with TripAdvisor's full rating distribution.

Reviews were collected with TripAdvisor's rating filters, so the sample over-represents low ratings.
data/raw/TripAdvisor_rating_counts.xlsx holds each attraction's full public rating histogram
(all languages; Excellent..Terrible = 5..1 stars), recorded at collection time.

Outputs (results/population/):
  population_sentiment_by_attraction.csv / _by_category.csv  - sentiment from the full rating histograms
  sample_vs_population.csv                                    - how far the sample departs from the population
  topic_prevalence_reweighted.csv                             - complaint-topic shares reweighted to the population
  population_results.json                                     - chi-square/Cramér's V on population counts, etc.

Reweighting: within each attraction, a review with s stars gets weight N_pop(a,s) / n_sample(a,s), i.e.
post-stratification on (attraction, star rating). Star->sentiment mapping follows the labelling protocol.
"""
import json
import os
import unicodedata

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results/population")
os.makedirs(OUT, exist_ok=True)


def norm(s):
    s = str(s).replace("ı", "i").replace("İ", "I")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return s.replace("pastenesi", "pastanesi").strip()


d = pd.read_csv(os.path.join(ROOT, "data/processed/reviews_publication_final.csv"))
t = pd.read_excel(os.path.join(ROOT, "data/raw/TripAdvisor_rating_counts.xlsx"))
t.columns = ["Attraction", "s5", "s4", "s3", "s2", "s1"]
t["key"] = t.Attraction.map(norm)
d["key"] = d.Attraction_Name.map(norm)
assert set(t.key) == set(d.key), set(t.key) ^ set(d.key)
cat = d.groupby("key").Attraction_Category.first()
name = d.groupby("key").Attraction_Name.first()
t["Category"] = t.key.map(cat)
t["Attraction_Name"] = t.key.map(name)
t["N_pop"] = t[["s1", "s2", "s3", "s4", "s5"]].sum(axis=1)
t["pop_neg"] = (t.s1 + t.s2) / t.N_pop * 100
t["pop_neu"] = t.s3 / t.N_pop * 100
t["pop_pos"] = (t.s4 + t.s5) / t.N_pop * 100
t["pop_mean_stars"] = (t.s1 + 2 * t.s2 + 3 * t.s3 + 4 * t.s4 + 5 * t.s5) / t.N_pop

samp = d.groupby("key").agg(n_sample=("Review_ID", "size"), sample_mean_stars=("Star_Rating", "mean"),
                            sample_neg=("Final_Sentiment", lambda s: (s == "Negative").mean() * 100),
                            sample_neu=("Final_Sentiment", lambda s: (s == "Neutral").mean() * 100),
                            sample_pos=("Final_Sentiment", lambda s: (s == "Positive").mean() * 100))
A = t.merge(samp, left_on="key", right_index=True)
A["net_pop"] = A.pop_pos - A.pop_neg
A["net_sample"] = A.sample_pos - A.sample_neg
A["neg_enrichment_factor"] = A.sample_neg / A.pop_neg.replace(0, np.nan)
cols = ["Category", "Attraction_Name", "N_pop", "n_sample", "pop_mean_stars", "sample_mean_stars", "pop_pos", "pop_neu",
        "pop_neg", "sample_pos", "sample_neu", "sample_neg", "net_pop", "net_sample", "neg_enrichment_factor"]
A[cols].sort_values("net_pop", ascending=False).round(2).to_csv(os.path.join(OUT, "sample_vs_population.csv"), index=False)
A[["Category", "Attraction_Name", "N_pop", "s5", "s4", "s3", "s2", "s1", "pop_mean_stars", "pop_pos", "pop_neu", "pop_neg",
   "net_pop"]].sort_values("net_pop", ascending=False).round(2).to_csv(
    os.path.join(OUT, "population_sentiment_by_attraction.csv"), index=False)

C = t.groupby("Category")[["s1", "s2", "s3", "s4", "s5", "N_pop"]].sum()
C["pop_neg"] = (C.s1 + C.s2) / C.N_pop * 100
C["pop_neu"] = C.s3 / C.N_pop * 100
C["pop_pos"] = (C.s4 + C.s5) / C.N_pop * 100
C["pop_mean_stars"] = (C.s1 + 2 * C.s2 + 3 * C.s3 + 4 * C.s4 + 5 * C.s5) / C.N_pop
C["net_pop"] = C.pop_pos - C.pop_neg
C["n_attractions"] = t.groupby("Category").size()
C.sort_values("net_pop", ascending=False).round(2).to_csv(os.path.join(OUT, "population_sentiment_by_category.csv"))

ct = pd.DataFrame({"Negative": C.s1 + C.s2, "Neutral": C.s3, "Positive": C.s4 + C.s5})
chi2, p, dof, exp = chi2_contingency(ct)
cramer = float(np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1))))
resid = ((ct - exp) / np.sqrt(exp)).round(1)

# ---- reweighted topic prevalence (Negative + Neutral reviews)
ta = pd.read_csv(os.path.join(ROOT, "results/tourism/review_topic_assignments.csv"))
labels = json.load(open(os.path.join(ROOT, "results/tourism/topic_labels.json")))
ta = ta.merge(d[["Review_ID", "Star_Rating", "key"]], on="Review_ID")
n_as = d.groupby(["key", "Star_Rating"]).size()
popc = t.set_index("key")[["s1", "s2", "s3", "s4", "s5"]]
ta["w"] = [popc.loc[k, f"s{s}"] / n_as.loc[(k, s)] for k, s in zip(ta.key, ta.Star_Rating)]
ta = ta[ta.topic >= 0]
ta["Topic"] = ta.topic.astype(str).map(lambda x: labels[x].replace("\n", " "))
rw = (ta.groupby(["Attraction_Category", "Topic"]).w.sum() / ta.groupby("Attraction_Category").w.sum() * 100).unstack().round(1)
un = (ta.groupby(["Attraction_Category", "Topic"]).size() / ta.groupby("Attraction_Category").size() * 100).unstack().round(1)
overall_rw = (ta.groupby("Topic").w.sum() / ta.w.sum() * 100).round(1)
overall_un = (ta.groupby("Topic").size() / len(ta) * 100).round(1)
rw.to_csv(os.path.join(OUT, "topic_prevalence_reweighted_by_category.csv"))
pd.DataFrame({"sample_share_pct": overall_un, "reweighted_share_pct": overall_rw}).to_csv(
    os.path.join(OUT, "topic_prevalence_overall.csv"))

res = {"N_population_reviews": int(t.N_pop.sum()), "n_sample": int(len(d)),
       "population_overall_pct": {"Positive": float((t.s4.sum() + t.s5.sum()) / t.N_pop.sum() * 100),
                                  "Neutral": float(t.s3.sum() / t.N_pop.sum() * 100),
                                  "Negative": float((t.s1.sum() + t.s2.sum()) / t.N_pop.sum() * 100)},
       "sample_overall_pct": (d.Final_Sentiment.value_counts(normalize=True) * 100).round(2).to_dict(),
       "category_x_sentiment_population": {"chi2": float(chi2), "dof": int(dof), "p": float(p), "cramers_v": cramer,
                                           "std_residuals": resid.to_dict(orient="index")},
       "rank_correlation_net_sentiment_sample_vs_population": float(
           A[["net_pop", "net_sample"]].corr(method="spearman").iloc[0, 1]),
       "topic_overall_sample_vs_reweighted": {"sample": overall_un.to_dict(), "reweighted": overall_rw.to_dict()},
       "note": "TripAdvisor counts cover all languages and all years; the sample is English(-translated) text."}
json.dump(res, open(os.path.join(OUT, "population_results.json"), "w"), indent=1, ensure_ascii=False)
pd.set_option("display.width", 250)
print(A[cols].sort_values("net_pop", ascending=False).round(1).to_string(index=False))
print(C.sort_values("net_pop", ascending=False).round(1).to_string())
print(json.dumps({k: v for k, v in res.items() if k != "category_x_sentiment_population"}, indent=1, ensure_ascii=False))
print("chi2", round(chi2, 1), "V", round(cramer, 3)); print(resid)
print("reweighted topic shares by category:\n", rw.to_string()); print("unweighted:\n", un.to_string())
