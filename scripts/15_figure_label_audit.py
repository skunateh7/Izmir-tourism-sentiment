import json, sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
r = json.load(open(os.path.join(ROOT, "results/label_comparison/label_comparison.json")))["vs_consensus"]
panels = [("Star", "a  Star-derived labels"), ("VADER_standard", "b  VADER, standard thresholds"), ("DistilBERT_CV", "c  DistilBERT, out-of-fold")]
labs = ["Negative", "Neutral", "Positive"]
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 8})
W = 174 / 25.4
fig, axes = plt.subplots(1, 3, figsize=(W, 2.75), constrained_layout=True)
norm = Normalize(0, 100)
for ax, (k, title) in zip(axes, panels):
    cm = np.array(r[k]["confusion_rows_human_cols_method"], float)
    pct = cm / cm.sum(1, keepdims=True) * 100
    im = ax.imshow(pct, cmap="Blues", norm=norm)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{int(cm[i, j])}\n{pct[i, j]:.0f}%", ha="center", va="center", fontsize=8,
                    color="white" if pct[i, j] > 55 else "#222222")
    ax.set_xticks(range(3), labs)
    ax.set_yticks(range(3), [f"{l}\n(n = {int(cm[i].sum())})" for i, l in enumerate(labs)])
    ax.set_title(title, fontsize=8.5, loc="left", color="#222222")
    ax.set_xlabel("Label assigned by method", color="#333333")
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0, colors="#333333")
axes[0].set_ylabel("Consensus human label", color="#333333")
cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.01)
cb.set_label("Row percentage (%)", color="#333333"); cb.outline.set_visible(False); cb.ax.tick_params(length=0, colors="#333333")
out = os.path.join(ROOT, "results/paper/figures/fig_label_audit.png"); fig.savefig(out, dpi=600)
fig.savefig(out.replace(".png", ".pdf"))
