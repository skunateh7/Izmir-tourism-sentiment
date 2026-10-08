"""
Step 15 — Sensitivity analyses requested in review (round 2).

(a) Text-based destination estimate. Post-stratification on star histograms estimates the share of positive
    RATINGS. To estimate the share of positive TEXT, the human audit (141 random non-override reviews, two
    annotators) gives P(text sentiment | rating group); applying it to each attraction's full star histogram gives
    a text-based estimate. Assumes P(text | rating group) is the same across attractions (audit too small to vary
    it). Bootstrap over audit items, 2000 resamples.
(b) Complaint themes restricted to Negative reviews: (i) shares of the main four topics among Negative-labelled
    reviews only; (ii) NMF k=4 refitted on Negative reviews only, matched to the main topics (Hungarian, cosine of
    term loadings).
(c) Stability of the topic solution: k = 3..6, 20 random NMF initialisations each; matched-topic cosine and
    document-assignment ARI against the reported (nndsvda) solution; c_v coherence across initialisations.
(d) Within-rating selection: topic shares by star level, and bootstrap CIs for the post-stratified topic shares
    resampling reviews within (attraction, star) strata.
Outputs: results/review2/*.json|csv
"""
import json
import os
import unicodedata

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results/review2")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)
B = 2000
d = pd.read_csv(os.path.join(ROOT, "data/processed/reviews_publication_final.csv"))
res = {}


def norm(s):
    s = str(s).replace("ı", "i").replace("İ", "I")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return s.replace("pastenesi", "pastanesi").strip()


GRP = {1: "Negative", 2: "Negative", 3: "Neutral", 4: "Positive", 5: "Positive"}
LABS = ["Negative", "Neutral", "Positive"]

# ======================= (a) text-based destination estimate
A = os.path.join(ROOT, "annotation")
key = pd.read_csv(os.path.join(A, "annotation_key_DO_NOT_SHARE.csv"))
sa = pd.read_excel(os.path.join(A, "completed/annotation_sheet_A.xlsx")).iloc[:, [0, 2]]
sb = pd.read_excel(os.path.join(A, "completed/annotation_sheet_B.xlsx")).iloc[:, [0, 2]]
sa.columns = ["Item", "A"]; sb.columns = ["Item", "B"]
m = key.merge(sa, on="Item").merge(sb, on="Item").dropna(subset=["A", "B"])
for c in ["A", "B"]:
    m[c] = m[c].str.strip().str.capitalize()
m = m[(m.Manual_Override == 0) & (m.Review_ID != 420)].reset_index(drop=True)
assert len(m) == 141, len(m)
m["G"] = m.Star_Rating.map(GRP)

t = pd.read_excel(os.path.join(ROOT, "data/raw/TripAdvisor_rating_counts.xlsx"))
t.columns = ["Attraction", "s5", "s4", "s3", "s2", "s1"]
t["key"] = t.Attraction.map(norm)
d["key"] = d.Attraction_Name.map(norm)
t["Category"] = t.key.map(d.groupby("key").Attraction_Category.first())
t["Attraction_Name"] = t.key.map(d.groupby("key").Attraction_Name.first())
pop = pd.DataFrame({"Negative": t.s1 + t.s2, "Neutral": t.s3, "Positive": t.s4 + t.s5})
pop.index = t.Attraction_Name


def cond(mm, col):
    """P(text label | rating group) as 3x3 DataFrame rows=rating group."""
    return pd.crosstab(mm.G, mm[col]).reindex(index=LABS, columns=LABS, fill_value=0).pipe(
        lambda x: x.div(x.sum(axis=1), axis=0))


def text_est(P, popc):
    return popc.values @ P.values / popc.values.sum()  # shares of text Negative/Neutral/Positive


def estimates(mm):
    out = {}
    for src in ["A", "B", "agreed"]:
        if src == "agreed":
            mmm = mm[mm.A == mm.B].assign(L=lambda x: x.A); col = "L"
        else:
            mmm, col = mm, src
        P = cond(mmm, col)
        out[src] = {"overall": text_est(P, pop.sum().to_frame().T),
                    "by_category": {c: text_est(P, g.sum().to_frame().T)
                                    for c, g in pop.groupby(t.set_index("Attraction_Name").Category)},
                    "by_attraction": {a: text_est(P, pop.loc[[a]]) for a in pop.index}}
    return out


point = estimates(m)
boots = [estimates(m.iloc[rng.integers(0, len(m), len(m))].reset_index(drop=True)) for _ in range(B)]


def ci(getter):
    v = np.array([getter(b) for b in boots]); return np.nanpercentile(v, [2.5, 97.5], axis=0)


rating_overall = (pop.sum() / pop.values.sum()).values
rows = []
for src in ["A", "B", "agreed"]:
    lo, hi = ci(lambda b: b[src]["overall"])
    rows.append({"level": "Overall", "unit": "All 14 entities", "label_source": src,
                 **{f"rating_{l}": rating_overall[i] * 100 for i, l in enumerate(LABS)},
                 **{f"text_{l}": point[src]["overall"].ravel()[i] * 100 for i, l in enumerate(LABS)},
                 **{f"text_{l}_lo": lo.ravel()[i] * 100 for i, l in enumerate(LABS)},
                 **{f"text_{l}_hi": hi.ravel()[i] * 100 for i, l in enumerate(LABS)}})
    for lvl, dct in [("Category", "by_category"), ("Attraction", "by_attraction")]:
        for u, est in point[src][dct].items():
            lo, hi = ci(lambda b: b[src][dct][u])
            pc = (pop.groupby(t.set_index("Attraction_Name").Category).sum().loc[u] if lvl == "Category" else pop.loc[u])
            rr = (pc / pc.sum()).values
            rows.append({"level": lvl, "unit": u, "label_source": src,
                         **{f"rating_{l}": rr[i] * 100 for i, l in enumerate(LABS)},
                         **{f"text_{l}": est.ravel()[i] * 100 for i, l in enumerate(LABS)},
                         **{f"text_{l}_lo": lo.ravel()[i] * 100 for i, l in enumerate(LABS)},
                         **{f"text_{l}_hi": hi.ravel()[i] * 100 for i, l in enumerate(LABS)}})
TE = pd.DataFrame(rows).round(1)
TE.to_csv(os.path.join(OUT, "text_based_population_estimates.csv"), index=False)
# rank agreement between rating-based and text-based net sentiment across attractions
ta_ = TE[(TE.level == "Attraction") & (TE.label_source == "agreed")]
rho = pd.Series(ta_.rating_Positive - ta_.rating_Negative).corr(pd.Series(ta_.text_Positive - ta_.text_Negative),
                                                                  method="spearman")
res["a_text_based_estimate"] = {
    "audit_n": int(len(m)), "agreed_n": int((m.A == m.B).sum()),
    "P_text_given_rating_agreed": cond(m[m.A == m.B].assign(L=lambda x: x.A), "L").round(3).to_dict(orient="index"),
    "overall": TE[TE.level == "Overall"].to_dict(orient="records"),
    "spearman_rating_vs_text_net_by_attraction": float(rho),
    "assumption": "P(text sentiment | rating group) pooled across attractions"}

# ======================= topic machinery (identical preprocessing to step 10)
src10 = open(os.path.join(ROOT, "scripts/10_tourism_and_topics.py")).read()
seg = src10.split("# ---------------- topic modelling on Negative + Neutral reviews\n")[1].split("nn_df = ")[0]
ns = {"re": __import__("re"), "ENGLISH_STOP_WORDS": __import__("sklearn.feature_extraction.text",
      fromlist=["x"]).ENGLISH_STOP_WORDS}
exec(seg, ns)  # DOMAIN_STOP, EVALUATIVE, STOP, tok — identical to step 10
tok = ns["tok"]
nn_df = d[d.Final_Sentiment.isin(["Negative", "Neutral"])].reset_index(drop=True)
docs = [" ".join(tok(x)) for x in nn_df.Original_Review_Text]
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.5, sublinear_tf=True)
X = vec.fit_transform(docs)
terms = np.array(vec.get_feature_names_out())
tokens = [x.split() for x in docs]
dic = Dictionary(tokens)
labels = {int(k): v.replace("\n", " ") for k, v in json.load(open(os.path.join(ROOT, "results/tourism/topic_labels.json"))).items()}
ref = NMF(4, init="nndsvda", random_state=42, max_iter=1000).fit(X)
Wref = ref.transform(X)
ref_assign = np.where(Wref.max(1) == 0, -1, Wref.argmax(1))
ta = pd.read_csv(os.path.join(ROOT, "results/tourism/review_topic_assignments.csv"))
assert (ta.topic.values == ref_assign).all(), "reference solution does not reproduce step 10"


def cos_match(Ca, Cb):
    na = Ca / np.linalg.norm(Ca, axis=1, keepdims=True); nb = Cb / np.linalg.norm(Cb, axis=1, keepdims=True)
    S = na @ nb.T; r, c = linear_sum_assignment(-S); return S, r, c


def cv_of(nmf):
    tops = [[w for w in terms[np.argsort(c)[::-1]] if " " not in w][:10] for c in nmf.components_]
    return CoherenceModel(topics=tops, texts=tokens, dictionary=dic, coherence="c_v", topn=10, processes=1).get_coherence()


# ======================= (b) Negative-only
neg = (nn_df.Final_Sentiment == "Negative").values
sh_all = pd.Series(ref_assign[ref_assign >= 0]).map(labels).value_counts(normalize=True) * 100
sh_neg = pd.Series(ref_assign[neg & (ref_assign >= 0)]).map(labels).value_counts(normalize=True) * 100
sh_neu = pd.Series(ref_assign[~neg & (ref_assign >= 0)]).map(labels).value_counts(normalize=True) * 100
Xn = X[neg]
keep = np.asarray((Xn > 0).sum(0)).ravel() >= 3
neg_fit = NMF(4, init="nndsvda", random_state=42, max_iter=1000).fit(Xn[:, keep])
Cref = ref.components_[:, keep]
S, r, c = cos_match(Cref, neg_fit.components_)
Wn = neg_fit.transform(Xn[:, keep]); neg_assign = Wn.argmax(1)
remap = {cc: rr for rr, cc in zip(r, c)}
neg_assign_m = np.array([remap[a] for a in neg_assign])
agree_neg = float((neg_assign_m == ref_assign[neg]).mean())
neg_topics = []
for rr, cc in zip(r, c):
    neg_topics.append({"main_topic": labels[rr], "cosine_with_main": round(float(S[rr, cc]), 3),
                       "top_terms_negative_only": list(terms[keep][np.argsort(neg_fit.components_[cc])[::-1][:10]]),
                       "share_pct": round(float((neg_assign_m == rr).mean() * 100), 1)})
res["b_negative_only"] = {
    "n_negative": int(neg.sum()), "n_neutral": int((~neg).sum()),
    "main_topic_shares_pct": pd.DataFrame({"Negative+Neutral": sh_all, "Negative only": sh_neg,
                                           "Neutral only": sh_neu}).round(1).to_dict(orient="index"),
    "refit_k4_negative_only": neg_topics,
    "refit_assignment_agreement_with_main": round(agree_neg, 3),
    "refit_ari_with_main": round(float(adjusted_rand_score(ref_assign[neg], neg_assign_m)), 3)}
# coherence curve on Negative only
Cn = []
dn = Dictionary([tokens[i] for i in np.where(neg)[0]])
for k in range(3, 11):
    f = NMF(k, init="nndsvda", random_state=42, max_iter=1000).fit(Xn[:, keep])
    tops = [[w for w in terms[keep][np.argsort(cc)[::-1]] if " " not in w][:10] for cc in f.components_]
    Cn.append({"k": k, "c_v": CoherenceModel(topics=tops, texts=[tokens[i] for i in np.where(neg)[0]], dictionary=dn,
                                             coherence="c_v", topn=10, processes=1).get_coherence()})
res["b_negative_only"]["coherence_negative_only"] = pd.DataFrame(Cn).round(4).to_dict(orient="records")

# ======================= (c) stability across initialisations
SEEDS = range(20)


def run(k, s):
    f = NMF(k, init="random", random_state=s, max_iter=2000).fit(X)
    W = f.transform(X); a = W.argmax(1)
    out = {"k": k, "seed": s, "c_v": cv_of(f), "assign": a.tolist(), "comp": f.components_}
    return out


runs = Parallel(n_jobs=-1)(delayed(run)(k, s) for k in range(3, 11) for s in SEEDS)
stab = []
for k in range(3, 11):
    R = [x for x in runs if x["k"] == k]
    base = NMF(k, init="nndsvda", random_state=42, max_iter=1000).fit(X)
    bassign = base.transform(X).argmax(1)
    pair_ari = [adjusted_rand_score(R[i]["assign"], R[j]["assign"]) for i in range(len(R)) for j in range(i + 1, len(R))]
    ref_cos, ref_ari = [], []
    per_topic = np.zeros((len(R), k))
    for i, x in enumerate(R):
        S, rr, cc = cos_match(base.components_, x["comp"])
        ref_cos.append(S[rr, cc].mean()); per_topic[i] = S[rr, cc]
        ref_ari.append(adjusted_rand_score(bassign, x["assign"]))
    row = {"k": k, "c_v_reference": cv_of(base), "c_v_mean": np.mean([x["c_v"] for x in R]),
           "c_v_sd": np.std([x["c_v"] for x in R], ddof=1), "c_v_min": np.min([x["c_v"] for x in R]),
           "c_v_max": np.max([x["c_v"] for x in R]),
           "matched_cosine_vs_reference_mean": np.mean(ref_cos), "matched_cosine_vs_reference_min": np.min(ref_cos),
           "ari_vs_reference_mean": np.mean(ref_ari), "ari_vs_reference_min": np.min(ref_ari),
           "pairwise_ari_mean": np.mean(pair_ari)}
    if k == 4:
        row["per_topic_min_cosine"] = {labels[j]: round(float(per_topic[:, j].min()), 3) for j in range(4)}
        row["n_runs_all_topics_cos_ge_0.9"] = int((per_topic.min(1) >= 0.9).sum())
        row["per_topic_mean_cosine"] = {labels[j]: round(float(per_topic[:, j].mean()), 3) for j in range(4)}
    stab.append(row)
ST = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, dict)} for r in stab]).round(3)
ST.to_csv(os.path.join(OUT, "topic_stability.csv"), index=False)
res["c_stability"] = {"n_inits_per_k": len(SEEDS), "init": "random", "rows": json.loads(ST.to_json(orient="records")),
                      "k4_per_topic": {kk: stab[1][kk] for kk in ["per_topic_min_cosine", "per_topic_mean_cosine",
                                                                  "n_runs_all_topics_cos_ge_0.9"]}}

# ======================= (d) within-rating selection
ta = ta.merge(d[["Review_ID", "Star_Rating", "key"]], on="Review_ID")
ta = ta[ta.topic >= 0].copy()
ta["Topic"] = ta.topic.map(labels)
t3 = ta[ta.Star_Rating <= 3]  # two 4-star reviews (overrides labelled Neutral) excluded from this table
by_star = (pd.crosstab(t3.Star_Rating, t3.Topic, normalize="index") * 100).round(1)
by_star["n"] = t3.Star_Rating.value_counts().sort_index()
by_star.to_csv(os.path.join(OUT, "topic_shares_by_star.csv"))
from scipy.stats import chi2_contingency  # noqa: E402
chi = chi2_contingency(pd.crosstab(t3.Star_Rating, t3.Topic))
n_as = d.groupby(["key", "Star_Rating"]).size()
popc = t.set_index("key")[["s1", "s2", "s3", "s4", "s5"]]
ta["w"] = [popc.loc[k, f"s{s}"] / n_as.loc[(k, s)] for k, s in zip(ta.key, ta.Star_Rating)]


def rw_shares(df):
    return df.groupby("Topic").w.sum() / df.w.sum() * 100


pt = rw_shares(ta)
strata = [g for _, g in ta.groupby(["key", "Star_Rating"])]
bs = []
for _ in range(B):
    bs.append(rw_shares(pd.concat([g.sample(len(g), replace=True, random_state=int(rng.integers(1e9))) for g in strata])))
bs = pd.DataFrame(bs)
RW = pd.DataFrame({"unweighted": (ta.Topic.value_counts(normalize=True) * 100), "reweighted": pt,
                   "rw_lo": bs.quantile(0.025), "rw_hi": bs.quantile(0.975)}).round(1)
RW.to_csv(os.path.join(OUT, "topic_reweighted_bootstrap.csv"))
res["d_within_rating"] = {"topic_shares_by_star": by_star.to_dict(orient="index"),
                          "chi2_star_x_topic": {"chi2": float(chi[0]), "dof": int(chi[2]), "p": float(chi[1])},
                          "reweighted_with_stratum_bootstrap_CI": RW.to_dict(orient="index")}


# ======================= (e) review length (very long multi-paragraph reviews were skipped during collection)
tl = ta.merge(d[["Review_ID", "Word_Count"]], on="Review_ID")
tl["above_stratum_median"] = tl.Word_Count > tl.groupby(["key", "Star_Rating"]).Word_Count.transform("median")
tl["top_quartile"] = tl.Word_Count >= tl.Word_Count.quantile(0.75)
le = {}
for col in ["above_stratum_median", "top_quartile"]:
    ct = pd.crosstab(tl[col], tl.Topic)
    c = chi2_contingency(ct)
    le[col] = {"shares_pct": (pd.crosstab(tl[col], tl.Topic, normalize="index") * 100).round(1).to_dict(orient="index"),
               "n": tl[col].value_counts().to_dict(), "chi2": float(c[0]), "dof": int(c[2]), "p": float(c[1])}
le["word_count_quartiles_all_reviews"] = d.Word_Count.quantile([0, .25, .5, .75, 1]).to_dict()
res["e_review_length"] = le
# per-entity x star counts: sample and TripAdvisor histogram (Supplementary Table S11)
cnt = d.groupby(["key", "Star_Rating"]).size().unstack(fill_value=0).reindex(columns=[1, 2, 3, 4, 5], fill_value=0)
S11 = pd.DataFrame({"Entity": t.set_index("key").Attraction_Name, "Category": t.set_index("key").Category})
for st_ in [1, 2, 3, 4, 5]:
    S11[f"{st_}-star sample / TripAdvisor"] = [f"{cnt.loc[k, st_]} / {t.set_index('key').loc[k, f's{st_}']}" for k in S11.index]
S11["Sample total"] = cnt.sum(axis=1).reindex(S11.index)
S11.sort_values(["Category", "Entity"]).to_csv(os.path.join(OUT, "collection_counts_by_entity_star.csv"), index=False)
json.dump(res, open(os.path.join(OUT, "review2_results.json"), "w"), indent=1, ensure_ascii=False, default=float)
pd.set_option("display.width", 250)
print(TE[TE.level != "Attraction"].to_string(index=False))
print(json.dumps({k: v for k, v in res.items() if k != "a_text_based_estimate"}, indent=1, ensure_ascii=False, default=float))
print(json.dumps(res["a_text_based_estimate"]["P_text_given_rating_agreed"], indent=1), "rho", rho)

# ======================= (f) test-set vs pooled cross-validated macro-F1 (Supplementary Table S13)
F = json.load(open(os.path.join(ROOT, "results/final/final_results.json")))["details"]
M = pd.read_csv(os.path.join(ROOT, "results/final/table_main_results.csv")).set_index("Model")
CV = pd.read_csv(os.path.join(ROOT, "results/generalisation/cv_comparison.csv"))
G = pd.read_csv(os.path.join(ROOT, "results/final/table_config_grid.csv"))
cvs = lambda s, f: CV[(CV.scheme == s) & (CV.family == f)].macro_f1.iloc[0]
rows = []
for f in ["VADER", "TFIDF-LR", "TFIDF-SVM", "CNN", "BiLSTM", "DistilBERT"]:
    rep = (G[(G.family == "VADER") & G.config.str.startswith("standard")].test_macro_f1.iloc[0] if f == "VADER"
           else F[f]["metrics"]["macro_f1"])  # VADER: standard thresholds, as in the CV experiment
    mean = M.loc[f, "Macro_F1"] if f in ["CNN", "BiLSTM", "DistilBERT"] else None
    r = cvs("random_5fold", f)
    rows.append(dict(family=f, test_rep=rep, test_ci=None if f == "VADER" else F[f]["macro_f1_ci95"], test_seed_mean=mean,
                     random_cv=r, loeo=cvs("leave_one_attraction_out", f), diff_rep=rep - r,
                     diff_mean=None if mean is None else mean - r))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "test_vs_cv.csv"), index=False)
