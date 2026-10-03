"""
Step 11 — Regenerate every manuscript figure and table from the frozen results (single source of truth).
Output: results/paper/figures/*.png (300 dpi) and results/paper/tables/*.csv
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda *p: os.path.join(ROOT, "results", *p)  # noqa: E731
FIG, TAB = R("paper/figures"), R("paper/tables")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)

# ---- palette (validated reference palette; sentiment = diverging red/gray/blue)
SENT = {"Negative": "#e34948", "Neutral": "#b9b8b2", "Positive": "#2a78d6"}
MODEL_C = {"VADER": "#eb6834", "TFIDF-LR": "#1baf7a", "TFIDF-SVM": "#008300", "BiLSTM": "#4a3aa7",
           "CNN": "#2a78d6", "DistilBERT": "#e87ba4"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
                     "font.size": 9, "pdf.fonttype": 42, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "legend.frameon": False})
NAMES = {"VADER": "VADER (val-tuned band)", "TFIDF-LR": "TF-IDF + LR", "TFIDF-SVM": "TF-IDF + SVM", "BiLSTM": "BiLSTM",
         "CNN": "Multi-kernel CNN", "DistilBERT": "DistilBERT", "VADER-std": "VADER (standard)"}
d = pd.read_csv(os.path.join(ROOT, "data/processed/reviews_publication_final.csv"))
CATS = ["History/Culture", "Nature", "Family", "Nightlife", "Beach/Coastal", "Food/Gastronomy"]
DISP = {"Family": "Accommodation"}  # display name used in the manuscript
disp = lambda c: DISP.get(c, c)  # noqa: E731


def save(fig, name):
    fig.savefig(os.path.join(FIG, name))
    os.makedirs(os.path.join(FIG, "vector"), exist_ok=True)
    fig.savefig(os.path.join(FIG, "vector", name.replace(".png", ".pdf")))
    plt.close(fig)


# ================= Table 1: dataset by attraction
pa = pd.read_csv(R("audit/per_attraction_counts.csv"))
pa.to_csv(os.path.join(TAB, "table1_dataset_by_attraction.csv"), index=False)

# ================= Figure 1: population (TripAdvisor) sentiment by category + sample vs population
PC = pd.read_csv(R("population/population_sentiment_by_category.csv")).set_index("Category").loc[CATS]
PJ = json.load(open(R("population/population_results.json")))
fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.5), gridspec_kw={"width_ratios": [1.6, 1], "wspace": 0.42})
left = np.zeros(len(PC))
for s, col in [("Negative", "pop_neg"), ("Neutral", "pop_neu"), ("Positive", "pop_pos")]:
    ax[0].barh(range(len(PC)), PC[col], left=left, color=SENT[s], label=s, height=0.62, edgecolor="white", linewidth=1.5)
    for i, (l, v) in enumerate(zip(left, PC[col])):
        if v >= 7:
            ax[0].text(l + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=7.5,
                       color="white" if s != "Neutral" else INK)
    left += PC[col].values
ax[0].set_yticks(range(len(PC)), [f"{disp(c)}  (N={int(n):,})" for c, n in zip(PC.index, PC.N_pop)])
ax[0].set_xlim(0, 100); ax[0].set_xlabel("Share of all TripAdvisor ratings (%)"); ax[0].grid(axis="y", visible=False)
ax[0].invert_yaxis(); ax[0].legend(ncol=3, loc="upper center", bbox_to_anchor=(0.45, -0.16))
ax[0].set_title("a", loc="left", fontweight="bold")
comp = pd.DataFrame({"Sample\n(N = 598)": PJ["sample_overall_pct"],
                     f"TripAdvisor\n(N = {PJ['N_population_reviews']:,})": PJ["population_overall_pct"]}).T
left = np.zeros(2)
for s in ["Negative", "Neutral", "Positive"]:
    ax[1].barh(range(2), comp[s], left=left, color=SENT[s], height=0.55, edgecolor="white", linewidth=1.5)
    for i, (l, v) in enumerate(zip(left, comp[s])):
        if v >= 6:
            ax[1].text(l + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=7.5,
                       color="white" if s != "Neutral" else INK)
    left += comp[s].values
ax[1].set_yticks(range(2), comp.index); ax[1].invert_yaxis(); ax[1].set_xlim(0, 100)
ax[1].set_xlabel("Share (%)"); ax[1].grid(axis="y", visible=False)
ax[1].set_title("b", loc="left", fontweight="bold")
save(fig, "fig1_dataset_overview.png")

# ================= Table 2 + Figure 2: main model comparison
M = pd.read_csv(R("final/table_main_results.csv"))
allruns = pd.read_csv(R("main/all_runs.csv"))
tpath = R("transformer/all_runs.csv")
if os.path.exists(tpath):
    allruns = pd.concat([allruns, pd.read_csv(tpath)])
vs = allruns[(allruns.family == "VADER") & allruns.config.str.startswith("standard")].iloc[0]
from src_metrics_shim import vader_std_row  # noqa: E402  (tiny helper generated below)
std = vader_std_row(vs, R)
M2 = pd.concat([pd.DataFrame([std]), M], ignore_index=True)
cols = ["Model", "Config", "n_runs", "Val_MacroF1", "Accuracy", "Accuracy_SD", "Macro_P", "Macro_R", "Macro_F1",
        "Macro_F1_SD", "Weighted_F1", "F1_Negative", "F1_Neutral", "F1_Positive", "Recall_Neutral",
        "Rep_MacroF1", "Rep_MacroF1_CI_lo", "Rep_MacroF1_CI_hi"]
M2[cols].round(4).to_csv(os.path.join(TAB, "table2_main_results.csv"), index=False)

order = list(M2.Model)
fig, ax = plt.subplots(figsize=(7.2, 3.3))
for i, m in enumerate(order):
    row = M2[M2.Model == m].iloc[0]
    base = m.replace("-std", "")
    c = MODEL_C.get(base, INK2)
    lo, hi = row.Rep_MacroF1_CI_lo, row.Rep_MacroF1_CI_hi
    ax.plot([lo, hi], [i, i], color=c, lw=2, solid_capstyle="round")
    ax.plot(row.Rep_MacroF1, i, "o", ms=8, color=c, mec="white", mew=1.5, zorder=3)
    seeds = allruns[(allruns.family == base) & (allruns.config == row.Config)].test_macro_f1
    if len(seeds) > 1:
        ax.plot(seeds, [i - 0.25] * len(seeds), "|", ms=7, color=c, alpha=0.8)
    ax.text(hi + 0.012, i, f"{row.Rep_MacroF1:.3f}" + (f"  (seed mean {row.Macro_F1:.3f} ± {row.Macro_F1_SD:.3f})"
                                                    if row.n_runs > 1 else ""), va="center", fontsize=7.5, color=INK2)
ax.set_yticks(range(len(order)), [NAMES.get(m, m) for m in order]); ax.invert_yaxis()
ax.set_xlim(0.3, 1.0); ax.set_xlabel("Test macro-F1 with bootstrap 95% CI (n = 90)")
ax.grid(axis="y", visible=False)
save(fig, "fig2_model_comparison.png")

# ================= Figure 3: confusion matrices (all selected models)
F = json.load(open(R("final/final_results.json")))
fams = [m for m in ["VADER", "TFIDF-SVM", "TFIDF-LR", "CNN", "BiLSTM", "DistilBERT"] if m in F["details"]]
fig, axs = plt.subplots(2, 3, figsize=(8.6, 6.2), gridspec_kw={"wspace": 0.35, "hspace": 0.45})
labs = ["Neg", "Neu", "Pos"]
for ax, f in zip(axs.ravel(), fams):
    cm = np.array(F["details"][f]["metrics"]["confusion_matrix"])
    pct = cm / cm.sum(1, keepdims=True)
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}\n{pct[i, j]*100:.0f}%", ha="center", va="center", fontsize=9.5,
                    color="white" if pct[i, j] > 0.55 else INK)
    ax.set_xticks(range(3), labs, fontsize=9); ax.set_yticks(range(3), labs, fontsize=9); ax.grid(False)
    ax.set_title(NAMES[f], fontsize=10, fontweight="bold"); ax.set_xlabel("Predicted", fontsize=9)
for ax in axs[:, 0]:
    ax.set_ylabel("True", fontsize=9)
save(fig, "fig3_confusion_matrices.png")

# ================= Figure 4 + Table 3: generalisation
G = pd.read_csv(R("generalisation/cv_comparison.csv"))
G.round(4).to_csv(os.path.join(TAB, "table3_generalisation.csv"), index=False)
fams_g = [f for f in ["VADER", "TFIDF-LR", "TFIDF-SVM", "BiLSTM", "CNN", "DistilBERT"] if f in set(G.family)]
fig, ax = plt.subplots(figsize=(7, 0.62 * len(fams_g) + 0.5))
for i, f in enumerate(fams_g):
    for k, (sch, mk, off) in enumerate([("random_5fold", "o", -0.14), ("leave_one_attraction_out", "s", 0.14)]):
        r = G[(G.family == f) & (G.scheme == sch)].iloc[0]
        c = MODEL_C[f]
        ax.plot([r.ci_lo, r.ci_hi], [i + off] * 2, color=c, lw=2, alpha=0.9 if k == 0 else 0.55)
        ax.plot(r.macro_f1, i + off, mk, ms=7, color=c if k == 0 else "white", mec=c, mew=1.8, zorder=3)
        ax.text(r.ci_hi + 0.01, i + off, f"{r.macro_f1:.3f}", va="center", fontsize=7, color=INK2)
ax.set_yticks(range(len(fams_g)), [("VADER (standard)" if f == "VADER" else NAMES[f]) for f in fams_g]); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
ax.set_xlabel("Pooled macro-F1 over all 598 reviews (bootstrap 95% CI)"); ax.set_xlim(min(0.35, G.ci_lo.min() - 0.02), max(0.85, G.ci_hi.max() + 0.06))
ax.plot([], [], "o", color=INK2, label="Random stratified 5-fold CV")
ax.plot([], [], "s", color="white", mec=INK2, mew=1.8, label="Leave-one-entity-out CV")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
save(fig, "fig4_generalisation.png")

# ================= Figure 5: attraction-level negative share, population vs sample
from statsmodels.stats.proportion import proportion_confint  # noqa: E402
A = pd.read_csv(R("population/sample_vs_population.csv")).sort_values("pop_neg")
P = pd.read_csv(R("population/population_sentiment_by_attraction.csv")).set_index("Attraction_Name")
lo, hi = [], []
for a_, r in A.iterrows():
    p_ = P.loc[r.Attraction_Name]; k = int(p_.s1 + p_.s2)
    l_, h_ = proportion_confint(k, int(p_.N_pop), method="wilson"); lo.append(l_ * 100); hi.append(h_ * 100)
fig, ax = plt.subplots(figsize=(7.4, 4.6))
y = np.arange(len(A))
ax.hlines(y, A.pop_neg, A.sample_neg, color=GRID, lw=2.5, zorder=1)
ax.hlines(y, lo, hi, color=SENT["Negative"], lw=2.2, zorder=2)
ax.plot(A.pop_neg, y, "o", color=SENT["Negative"], ms=7, mec="white", mew=1.2, zorder=3,
        label="All TripAdvisor ratings (Wilson 95% CI)")
ax.plot(A.sample_neg, y, "o", mfc="white", mec=SENT["Negative"], mew=1.6, ms=6.5, zorder=3, label="Collected sample")
ax.set_yticks(y, [f"{a}  [{disp(c)}; N={int(n):,}]" for a, c, n in zip(A.Attraction_Name, A.Category, A.N_pop)], fontsize=7.6)
ax.set_xlim(0, 70); ax.set_xlabel("Negative reviews, 1-2 stars (%)"); ax.grid(axis="y", visible=False)
ax.legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=2)
save(fig, "fig5_attraction_sentiment.png")
A.round(2).to_csv(os.path.join(TAB, "table4_attraction_sentiment.csv"), index=False)

# ================= Figure 6 + Table 5: topics
T = json.load(open(R("tourism/tourism_results.json")))["topic_model"]
labels = json.load(open(R("tourism/topic_labels.json"))) if os.path.exists(R("tourism/topic_labels.json")) else {}
coh = pd.read_csv(R("tourism/topic_coherence.csv"))
tb = pd.read_csv(R("tourism/topic_by_category_pct.csv"), index_col=0)
tb = tb[[c for c in tb.columns if c != "-1"]]
fig, ax = plt.subplots(2, 1, figsize=(7.6, 7.4), gridspec_kw={"height_ratios": [0.75, 1.5], "hspace": 0.45})
ax[0].plot(coh.k, coh.c_v, "-o", color=MODEL_C["CNN"], lw=2, ms=5, mec="white")
kb = T["selected_k"]
ax[0].plot(kb, coh.set_index("k").c_v[kb], "o", ms=10, mfc="none", mec=INK, mew=1.5)
ax[0].set_xlabel("Number of topics k"); ax[0].set_ylabel("Mean c_v coherence")
ax[0].set_title("a", loc="left", fontweight="bold")
tl = [labels.get(str(t), f"T{t}") for t in range(kb)]
im = ax[1].imshow(tb.values, cmap="Blues", vmin=0, vmax=100, aspect="auto")
for i in range(tb.shape[0]):
    for j in range(tb.shape[1]):
        ax[1].text(j, i, f"{tb.values[i, j]:.0f}", ha="center", va="center", fontsize=10,
                   color="white" if tb.values[i, j] > 55 else INK)
ax[1].set_xticks(range(tb.shape[1]), tl, fontsize=9); ax[1].set_yticks(range(tb.shape[0]), [disp(c) for c in tb.index], fontsize=9.5)
ax[1].grid(False)
ax[1].set_title("b", loc="left", fontweight="bold")
save(fig, "fig6_topics.png")
pd.DataFrame([{"Topic": labels.get(str(t["topic"]), f"T{t['topic']}"), "n_reviews": t["n_reviews"],
               "share_pct": round(t["share_pct"], 1), "pct_negative_within": round(t["pct_negative_within"], 1),
               "top_terms": ", ".join(t["top_terms"][:10]),
               "main_categories": "; ".join(f"{k} ({v})" for k, v in t["top_categories"].items())}
              for t in T["topics"]]).to_csv(os.path.join(TAB, "table5_topics.csv"), index=False)

# ================= Figure 7 + Table 6: ablation
if os.path.exists(R("ablation/ablation.csv")):
    Ab = pd.read_csv(R("ablation/ablation.csv"))
    Ab.round(4).to_csv(os.path.join(TAB, "table6_ablation.csv"), index=False)
    C = Ab[Ab.model == "CNN"].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    yy = np.arange(len(C))
    ax.errorbar(C.test_mean, yy + 0.12, xerr=C.test_sd, fmt="o", color=MODEL_C["CNN"], ms=6, capsize=0, lw=1.8,
                mec="white", label="Test (mean ± SD, 5 seeds)")
    ax.errorbar(C.val_mean, yy - 0.12, xerr=C.val_sd, fmt="s", color=MODEL_C["CNN"], mfc="white", ms=5.5,
                capsize=0, lw=1.2, alpha=0.8, label="Validation")
    ax.axvline(C.test_mean[0], color=INK2, lw=0.8, ls="--")
    ax.set_yticks(yy, C.variant); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlabel("CNN macro-F1"); ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    save(fig, "fig7_ablation.png")

# ================= Supplementary: training curves
Cv = json.load(open(R("main/training_curves.json")))
sel = F["selected_configs"]
fig, axs = plt.subplots(1, 2, figsize=(9, 3))
for ax, f in zip(axs, ["CNN", "BiLSTM"]):
    h = pd.DataFrame(Cv[f"{f}|{sel[f]}|42"])
    ax.plot(h.epoch, h.loss, color=MODEL_C[f], lw=2, label="train loss")
    ax.plot(h.epoch, h.val_loss, color=MODEL_C[f], lw=2, ls="--", label="val loss")
    a2 = ax.twinx() if False else None  # single axis only (no dual scales)
    ax.set_title(f"{NAMES[f]} ({sel[f]}, seed 42)", loc="left", fontsize=8.5, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-entropy loss"); ax.legend()
save(fig, "figS1_training_loss.png")
fig, axs = plt.subplots(1, 2, figsize=(9, 3))
for ax, f in zip(axs, ["CNN", "BiLSTM"]):
    h = pd.DataFrame(Cv[f"{f}|{sel[f]}|42"])
    ax.plot(h.epoch, h.acc, color=MODEL_C[f], lw=2, label="train accuracy")
    ax.plot(h.epoch, h.val_macro_f1, color=MODEL_C[f], lw=2, ls="--", label="val macro-F1")
    ax.set_ylim(0, 1.02); ax.set_xlabel("Epoch"); ax.set_ylabel("Score"); ax.legend()
    ax.set_title(f"{NAMES[f]} ({sel[f]}, seed 42)", loc="left", fontsize=8.5, fontweight="bold")
save(fig, "figS2_training_scores.png")

for t in ["table_pairwise_tests.csv", "table_config_grid.csv"]:
    pd.read_csv(R("final", t)).round(4).to_csv(os.path.join(TAB, "tableS_" + t.replace("table_", "")), index=False)
print("figures:", sorted(os.listdir(FIG))); print("tables:", sorted(os.listdir(TAB)))
