"""
Step 14 - Post-stratified share of the service-encounter and food-quality complaint themes, with review-level and
entity-level (cluster) bootstrap intervals (Table S18). Needs no review texts.
"""
import json, os, unicodedata
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def norm(s):
    s = str(s).replace("ı", "i").replace("İ", "I")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return s.replace("pastenesi", "pastanesi").strip()
d = pd.read_csv(os.path.join(ROOT, "data/processed/reviews_metadata.csv")); d["key"] = d.Attraction_Name.map(norm)
h = pd.read_excel(os.path.join(ROOT, "data/raw/TripAdvisor_rating_counts.xlsx")); h.columns = ["A", 5, 4, 3, 2, 1]
h["key"] = h.A.map(norm); h = h.set_index("key")
labels = json.load(open(os.path.join(ROOT, "results/tourism/topic_labels.json")))
t = pd.read_csv(os.path.join(ROOT, "results/tourism/review_topic_assignments.csv")).merge(d[["Review_ID", "Star_Rating", "key"]], on="Review_ID")
t = t[t.topic >= 0]
n = d.groupby(["key", "Star_Rating"]).size()
t["w"] = [h.loc[k, s] / n.loc[(k, s)] for k, s in zip(t.key, t.Star_Rating)]
name = t.topic.astype(str).map(lambda x: labels[x].replace("\n", " "))
keep = name.str.contains("Service encounter") | name.str.contains("Food quality")
t["keep"] = keep.values
stat = lambda x: x[x.keep].w.sum() / x.w.sum() * 100
rng = np.random.default_rng(1); ents = t.key.unique(); B = 5000
cl = [stat(pd.concat([t[t.key == e] for e in rng.choice(ents, len(ents), replace=True)])) for _ in range(B)]
rv = [stat(t.iloc[rng.integers(0, len(t), len(t))]) for _ in range(B)]
res = {"unweighted_share": round(float(t.keep.mean() * 100), 1), "post_stratified_share": round(float(stat(t)), 1),
       "review_level_ci": [round(float(np.percentile(rv, q)), 1) for q in (2.5, 97.5)],
       "entity_level_ci": [round(float(np.percentile(cl, q)), 1) for q in (2.5, 97.5)], "bootstrap_resamples": B}
os.makedirs(os.path.join(ROOT, "results/label_comparison"), exist_ok=True)
json.dump(res, open(os.path.join(ROOT, "results/label_comparison/topic_cluster_bootstrap.json"), "w"), indent=1)
print(res)
