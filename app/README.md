# Streamlit companion app

An interactive companion to the manuscript *From Tourist Reviews to Destination Intelligence: Pretraining,
Cross-Entity Generalisation and Rating–Text Disagreement in İzmir*. Every number and chart is read from the study's
`results/` folder, so the app always agrees with the paper. It contains no review texts.

| Page | Content |
|---|---|
| Home | Key figures and findings |
| Model comparison | Test-set (Table 3) and pooled cross-validated scores, confusion matrices, paired tests, 508-review sensitivity check (RQ1) |
| Generalisation | Random vs leave-one-entity-out CV, per-entity accuracy (RQ2) |
| Neutral class & labels | Blind label audit, error rates, error taxonomy, ablation |
| Destination sentiment | Category and entity sentiment from all 13,209 TripAdvisor ratings (RQ3) |
| Complaint themes | NMF topics, unweighted and rating-distribution-adjusted (RQ4) |
| Analyse a review | Type your own review (or upload a CSV) and see how each model classifies it |
| About | Methods, reproducibility, data and AI-use statements |

## Run locally

From the repository root:

```bash
pip install -r app/requirements.txt
streamlit run app/Home.py
```

## Deploy on Streamlit Community Cloud

1. Push the repository to GitHub.
2. On share.streamlit.io choose **New app**, select the repository and set the main file to `app/Home.py`.
   Streamlit installs `app/requirements.txt` automatically.

## Models used by "Analyse a review"

* **TF-IDF + SVM / TF-IDF + LR** (`app/models/*.joblib`) are refitted by `scripts/13_export_app_models.py`
  on the locked training split; the script checks that they reproduce the manuscript's test macro-F1
  (0.705 and 0.686). The files hold only the fitted vocabulary and weights. They require scikit-learn 1.8.0;
  if you use another version, rerun the script (it needs the full dataset).
* **VADER** uses the standard thresholds or the validation-tuned neutral band (−0.1, 0.7).
* **DistilBERT** (the study's best model) is optional because its weights (~260 MB) exceed GitHub's file limit:
  1. On Google Colab (GPU), run `python scripts/05_transformer_distilbert.py --export-app-model`. This fine-tunes the
     selected configuration (learning rate 2e-5, class-weighted, seed 42; test macro-F1 0.775 in the study, small
     GPU-dependent variation is normal) and saves it to `app/models/distilbert/`.
  2. Either keep that folder locally (it is picked up automatically), or upload it to a Hugging Face model repository
     and set the secret `DISTILBERT_MODEL = "your-user/your-model"` in Streamlit Cloud (or as an environment variable).
  3. Uncomment `torch` and `transformers<5` in `app/requirements.txt`.

The models are trained on 418 English TripAdvisor reviews. They are reliable for clearly positive or negative text but
not for individual neutral or mixed reviews; use them for aggregate trends.
