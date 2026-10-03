# İzmir tourist-review sentiment: code, results and Streamlit app

Code, results and interactive app for the manuscript
**"From Tourist Reviews to Destination Intelligence: Pretraining, Cross-Entity Generalisation and Mixed Sentiment
in İzmir"** (under review). Every table and figure in the paper is produced by the scripts in this repository,
and the app reads the same result files.

## Interactive app

```bash
pip install -r app/requirements.txt
streamlit run app/Home.py
```

It shows the model comparison, generalisation to unseen tourism entities, the label audit and error analysis,
destination-level sentiment and complaint themes, and lets you classify your own text.
See [`app/README.md`](app/README.md) for deployment and for adding the fine-tuned DistilBERT model.

## Study in brief

* 598 English TripAdvisor reviews of 14 tourism entities in İzmir Province (six categories), collected in
  September 2026, plus each entity's full rating histogram (13,209 ratings).
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
scripts/             numbered pipeline 01-13
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
```

## Licence

Code is released under the MIT Licence. Result files are provided for verification of the published findings.
