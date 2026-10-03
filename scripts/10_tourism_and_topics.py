"""
Step 10 — Destination-level sentiment analysis and complaint-topic modelling.

Descriptive/inferential (uses the human-validated labels, not model predictions):
  * sentiment distribution per category and attraction, Wilson 95% CIs for the negative share
  * chi-square test of independence (category × sentiment) + Cramér's V, standardised residuals
  * Kruskal-Wallis test of star ratings across categories
Topic modelling on Negative + Neutral reviews:
  * TF-IDF (uni+bi-grams) -> NMF, k = 3..10, k chosen by mean c_v coherence (gensim),
    NMF initialised with nndsvda (deterministic). Topic labels are assigned by the researchers
    from the top terms and the three most representative reviews (exported for checking).
"""
import json
import os
import re

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from scipy.stats import chi2_contingency, kruskal
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from statsmodels.stats.proportion import proportion_confint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results/tourism")
os.makedirs(OUT, exist_ok=True)
d = pd.read_csv(os.path.join(ROOT, "data/processed/reviews_publication_final.csv"))
res = {}

# ---------------- category / attraction sentiment
ct = pd.crosstab(d.Attraction_Category, d.Final_Sentiment)[["Negative", "Neutral", "Positive"]]
chi2, p, dof, exp = chi2_contingency(ct)
n = ct.values.sum()
cramer = float(np.sqrt(chi2 / (n * (min(ct.shape) - 1))))
resid = (ct - exp) / np.sqrt(exp)
res["category_x_sentiment"] = {"chi2": float(chi2), "dof": int(dof), "p": float(p), "cramers_v": cramer,
                               "min_expected_count": float(exp.min()),
                               "standardised_residuals": resid.round(2).to_dict(orient="index")}
groups = [g.Star_Rating.values for _, g in d.groupby("Attraction_Category")]
kw = kruskal(*groups)
res["stars_by_category_kruskal"] = {"H": float(kw.statistic), "p": float(kw.pvalue)}

def summarise(by):
    rows = []
    for k, g in d.groupby(by):
        nn = len(g); neg = (g.Final_Sentiment == "Negative").sum(); pos = (g.Final_Sentiment == "Positive").sum()
        lo, hi = proportion_confint(neg, nn, method="wilson")
        rows.append({**({"Category": g.Attraction_Category.iloc[0]} if by == "Attraction_Name" else {}),
                     by: k, "n": nn, "mean_stars": g.Star_Rating.mean(),
                     "pct_positive": pos / nn * 100, "pct_neutral": (g.Final_Sentiment == "Neutral").mean() * 100,
                     "pct_negative": neg / nn * 100, "neg_ci_lo": lo * 100, "neg_ci_hi": hi * 100,
                     "net_sentiment": (pos - neg) / nn * 100})
    return pd.DataFrame(rows).sort_values("net_sentiment", ascending=False)
summarise("Attraction_Category").to_csv(os.path.join(OUT, "sentiment_by_category.csv"), index=False)
summarise("Attraction_Name").to_csv(os.path.join(OUT, "sentiment_by_attraction.csv"), index=False)

# ---------------- topic modelling on Negative + Neutral reviews
DOMAIN_STOP = set("""izmir turkey turkish place places really just like went go going got get did does
also time visit visited one would could even us also bit lot much many well way make made told said
ilica cesme alacati alsancak ephesus marriott reyhan deniz tavaci recep usta scappi bios punt bostanli
kordon konak hotel restaurant beach bar pub park tower clock wildlife key pastanesi plaji meyhanesi
don didn doesn isn wasn aren weren ve ll""".split())
# Sentiment-bearing words (|VADER valence| >= 1.0) are removed so topics capture ASPECTS, not polarity.
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # noqa: E402
EVALUATIVE = {w for w, v in SentimentIntensityAnalyzer().lexicon.items() if abs(v) >= 1.0 and w.isalpha()}
STOP = set(ENGLISH_STOP_WORDS) | DOMAIN_STOP | EVALUATIVE
# negations kept out of topic vocab on purpose: topics describe WHAT is discussed, not polarity


def tok(t):
    t = re.sub(r"[‘’]", "'", str(t).lower())
    return [w for w in re.findall(r"[a-z]+", t) if len(w) > 2 and w not in STOP]


nn_df = d[d.Final_Sentiment.isin(["Negative", "Neutral"])].reset_index(drop=True)
docs = [" ".join(tok(t)) for t in nn_df.Original_Review_Text]
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.5, sublinear_tf=True)
X = vec.fit_transform(docs)
terms = np.array(vec.get_feature_names_out())
tokens = [doc.split() for doc in docs]
dic = Dictionary(tokens)

coh = []
models = {}
for k in range(3, 11):
    nmf = NMF(k, init="nndsvda", random_state=42, max_iter=1000).fit(X)
    tops = [[w for w in terms[np.argsort(c)[::-1]] if " " not in w][:10] for c in nmf.components_]
    c_v = CoherenceModel(topics=tops, texts=tokens, dictionary=dic, coherence="c_v", topn=10).get_coherence()
    coh.append({"k": k, "c_v": float(c_v), "reconstruction_err": float(nmf.reconstruction_err_)})
    models[k] = nmf
    print(f"k={k} c_v={c_v:.4f}", flush=True)
C = pd.DataFrame(coh)
C.to_csv(os.path.join(OUT, "topic_coherence.csv"), index=False)
K = int(C.sort_values("c_v", ascending=False).iloc[0].k)
nmf = models[K]
W = nmf.transform(X)
nn_df["topic"] = W.argmax(1)
nn_df["topic_weight"] = W.max(1)
nn_df.loc[nn_df.topic_weight == 0, "topic"] = -1

topics = []
for t in range(K):
    comp = nmf.components_[t]
    top_terms = list(terms[np.argsort(comp)[::-1][:12]])
    members = nn_df[nn_df.topic == t]
    reps = members.sort_values("topic_weight", ascending=False).head(3)
    topics.append({"topic": t, "top_terms": top_terms, "n_reviews": int(len(members)),
                   "share_pct": float(len(members) / len(nn_df) * 100),
                   "pct_negative_within": float((members.Final_Sentiment == "Negative").mean() * 100) if len(members) else 0,
                   "top_categories": members.Attraction_Category.value_counts().head(3).to_dict(),
                   "representative_review_ids": reps.Review_ID.tolist(),
                   "representative_excerpts": [r[:220] for r in reps.Original_Review_Text]})
pd.crosstab(nn_df.Attraction_Category, nn_df.topic, normalize="index").mul(100).round(1).to_csv(
    os.path.join(OUT, "topic_by_category_pct.csv"))
nn_df[["Review_ID", "Attraction_Name", "Attraction_Category", "Final_Sentiment", "topic", "topic_weight"]].to_csv(
    os.path.join(OUT, "review_topic_assignments.csv"), index=False)
res["topic_model"] = {"method": "NMF on TF-IDF (1-2 grams, min_df=3, max_df=0.5), Negative+Neutral reviews; "
                                "English stopwords, attraction/place names and VADER words with |valence|>=1 removed",
                      "n_evaluative_words_removed_from_lexicon": len(EVALUATIVE),
                      "n_documents": int(len(nn_df)), "k_range": [3, 10], "selected_k": K,
                      "selection_criterion": "max mean c_v coherence (top-10 unigrams)", "topics": topics}
json.dump(res, open(os.path.join(OUT, "tourism_results.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "topic_model"}, indent=1))
print("K =", K)
for t in topics:
    print(t["topic"], t["n_reviews"], round(t["pct_negative_within"]), t["top_terms"][:10], t["top_categories"])
