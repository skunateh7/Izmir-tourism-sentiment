# İzmir tourist-review sentiment: code, results and Streamlit app

Code, results and interactive app for the manuscript
**"Evaluating Sentiment Measurement Validity in Online Reviews: Evidence from Tourism"** (submitted to the
*Journal of Computational Social Science*). It extends the earlier analysis of the same corpus,
"From Tourist Reviews to Destination Intelligence: Pretraining, Cross-Entity Generalisation and Rating–Text
Disagreement in İzmir". Every table and figure in the paper is produced by the scripts in this repository,
and the app reads the same result files. The new label-comparison analyses are described in the section
"JCSS revision" below.

## Interactive app

**Live app:** https://izmirtourismsentiment.streamlit.app/

To run it locally:

```bash
pip install -r app/requirements.txt
streamlit run app/Home.py
```

It shows the model comparison, generalisation to unseen tourism entities, the label audit and error analysis,
destination-level sentiment and complaint themes, and lets you classify your own text.
See [`app/README.md`](app/README.md) for deployment and for adding the fine-tuned DistilBERT model.

## Study in brief

* 598 English TripAdvisor reviews of 14 tourism entities in İzmir Province (six categories), collected in
  August 2026, plus each entity's full rating histogram (13,209 ratings).
* Six approaches: VADER, TF-IDF + logistic regression, TF-IDF + linear SVM, multi-kernel CNN, BiLSTM and fine-tuned
  DistilBERT, compared with validation-only model selection, five seeds, bootstrap confidence intervals,
  McNemar tests, pooled random 5-fold and leave-one-entity-out cross-validation.
* DistilBERT performed best (pooled cross-validated macro-F1 0.705; 0.714 on unseen entities); models trained from
  scratch were no better than TF-IDF classifiers. Neutral (3-star) reviews were hardest, largely because many
  describe mixed experiences.

## Repository layout

```
app/                 Streamlit app (reads results/; no review texts)
data/processed/      reviews_metadata.csv (IDs, entity, category, stars, labels - no texts), split_main.json
data/raw/            TripAdvisor_rating_counts.xlsx (full rating histograms)
scripts/             numbered pipeline 01-16
src/                 preprocessing, models, metrics
results/             all outputs; results/paper/ = manuscript figures and tables
```

## Data availability

Because of TripAdvisor's terms of use, **full review texts are not included**. The repository provides the derived
labels, the locked data splits, all predictions and results, and a description of the collection procedure, which
support replication of the collection and analysis procedure. Running the pipeline from scratch (`run_all.sh`)
requires a review file with the columns of `reviews_metadata.csv` plus `Original_Review_Text` and
`Processed_Text`; the app and all result files work without it.

## Reproducing the analysis

```bash
pip install -r requirements.txt
bash run_all.sh                              # CPU steps (about 1.5 h)
python scripts/05_transformer_distilbert.py  # DistilBERT (GPU / Google Colab)
python scripts/05b_transformer_cv.py         # DistilBERT cross-validation (GPU)
python scripts/05c_distilbert_config_cv.py   # all four DistilBERT configurations + nested selection (GPU; then --summarise)
```

## Licence

Code is released under the MIT Licence. Result files are provided for verification of the published findings.

## JCSS revision: label comparison and robustness analyses

These files support *Evaluating Sentiment Measurement Validity in Online Reviews: Evidence from Tourism*
(Journal of Computational Social Science submission). They need **no review texts**.

| File | What it is |
|---|---|
| `annotation/audit_labels_blind.csv` | The 150 audit items: review ID, star rating, raw rating-derived label, final label, override flag, and both annotators' blind text-only labels (no texts). `Annotator_A` is the second annotator and `Annotator_B` is the author, as in `results/label_audit/label_agreement_main_vs_overrides.json` |
| `results/label_comparison/vader_compound_audited.csv` | VADER compound score for each audited review (lets the VADER labels be rebuilt without texts) |
| `scripts/13_label_comparison.py` | Table 2, Fig. 1 data, Tables S16, S16b and S17 → `results/label_comparison/label_comparison.json` |
| `scripts/14_topic_cluster_bootstrap.py` | Table S18 (post-stratified complaint-theme share, review- and entity-level bootstrap) |
| `scripts/15_figure_label_audit.py` | Fig. 1 → `results/paper/figures/fig_label_audit.png/.pdf` |

Run from the repository root:

```
python scripts/13_label_comparison.py
python scripts/14_topic_cluster_bootstrap.py
python scripts/15_figure_label_audit.py
```

Audit accounting: 150 selected → 1 management reply excluded → 8 manual overrides assessed separately →
141 main audit reviews (raw and final labels identical) → 121 on which both annotators agree.
