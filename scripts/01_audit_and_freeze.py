"""
Step 1 — Audit the cleaned review workbook and freeze the publication dataset.

Input : data/raw/Izmir_Tourist_Review_Cleaned.xlsx  (sheet 'Cleaned Reviews', 599 rows)
Output: data/processed/reviews_publication_final.csv   (frozen; never edit by hand)
        data/processed/split_main.json                   (stratified 70/15/15, seed 42)
        results/audit/audit_report.json, near_duplicates.csv, per_attraction_counts.csv
"""
import hashlib
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.preprocess import clean_text  # noqa: E402

RAW = os.path.join(ROOT, "data/raw/Izmir_Tourist_Review_Cleaned.xlsx")
OUT = os.path.join(ROOT, "data/processed")
AUD = os.path.join(ROOT, "results/audit")
os.makedirs(OUT, exist_ok=True)
os.makedirs(AUD, exist_ok=True)
SEED = 42

df = pd.read_excel(RAW, sheet_name="Cleaned Reviews")
report = {"source_file": os.path.basename(RAW), "rows_loaded": int(len(df))}

# ---------- basic integrity ----------
df["Original_Review_Text"] = df["Original_Review_Text"].astype(str)
norm = df["Original_Review_Text"].str.replace(r"\s+", " ", regex=True).str.strip().str.lower()
report["duplicate_ids"] = int(df["Review_ID"].duplicated().sum())
report["exact_duplicate_texts"] = int(norm.duplicated().sum())
report["missing_text"] = int((norm.isin(["", "nan"])).sum())
report["missing_rating"] = int(df["Star_Rating"].isna().sum())
report["ratings_out_of_range"] = int((~df["Star_Rating"].isin([1, 2, 3, 4, 5])).sum())

# ---------- near duplicates (char 3-5 gram TF-IDF cosine) ----------
tf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1).fit_transform(norm)
sim = cosine_similarity(tf)
np.fill_diagonal(sim, 0)
pairs = []
for i, j in zip(*np.where(np.triu(sim) >= 0.80)):
    pairs.append({"id_a": int(df.Review_ID[i]), "id_b": int(df.Review_ID[j]),
                  "attr_a": df.Attraction_Name[i], "attr_b": df.Attraction_Name[j],
                  "cosine": round(float(sim[i, j]), 3),
                  "text_a": df.Original_Review_Text[i][:160], "text_b": df.Original_Review_Text[j][:160]})
pd.DataFrame(pairs).to_csv(os.path.join(AUD, "near_duplicates.csv"), index=False)
report["near_duplicate_pairs_cos_ge_0.80"] = len(pairs)
report["max_pairwise_cosine"] = round(float(sim.max()), 3)

# ---------- attraction / category consistency ----------
ac = df.groupby("Attraction_Name")["Attraction_Category"].nunique()
report["attractions_with_multiple_categories"] = ac[ac > 1].index.tolist()
report["n_attractions"] = int(df.Attraction_Name.nunique())
report["n_categories"] = int(df.Attraction_Category.nunique())

# ---------- language composition (English function-word heuristic) ----------
EN = set("the and was is it to of a in for we with but not this very were you they are on at "
         "that have had there be our my i so if as all its one place food good great nice".split())
TR = set("ve bir bu çok ile için ama da de güzel gibi olarak daha en yok var".split())
def lang(t):
    w = re.findall(r"[a-zçğıöşü]+", t.lower())
    if not w:
        return "empty"
    en = sum(x in EN for x in w) / len(w)
    tr = sum(x in TR for x in w) / len(w) + (0.2 if re.search(r"[çğışöü]", t.lower()) and en < 0.15 else 0)
    return "en" if en >= 0.15 and en > tr else ("tr" if tr > en else "uncertain")
df["Language"] = df["Original_Review_Text"].apply(lang)
report["language_distribution"] = df["Language"].value_counts().to_dict()
report["non_ascii_reviews"] = int(df.Original_Review_Text.apply(lambda t: bool(re.search(r"[^\x00-\x7F]", t))).sum())

# Manual verification: the two 'uncertain' reviews (IDs 156, 397) were read and are English.
MANUAL_LANG = {156: "en", 397: "en"}
df["Language"] = [MANUAL_LANG.get(int(i), l) for i, l in zip(df.Review_ID, df.Language)]
report["language_distribution_after_manual_check"] = df["Language"].value_counts().to_dict()

# ---------- exclusions decided during audit (documented, not silent) ----------
EXCLUDE = {347: "Truncated duplicate of Review 346 (same tourist review captured twice; cosine 0.948)."}
report["excluded_reviews"] = {str(k): v for k, v in EXCLUDE.items()}
df = df[~df.Review_ID.isin(EXCLUDE)].reset_index(drop=True)

# ---------- labels ----------
def star2lab(s):
    return "Negative" if s <= 2 else ("Neutral" if s == 3 else "Positive")
df["Rating_Label"] = df["Star_Rating"].apply(star2lab)
df["Manual_Override"] = (df["Rating_Label"] != df["Final_Sentiment"]).astype(int)
report["manual_overrides"] = int(df.Manual_Override.sum())
report["override_ids"] = df.loc[df.Manual_Override == 1, "Review_ID"].astype(int).tolist()
report["label_distribution"] = df.Final_Sentiment.value_counts().to_dict()
report["star_distribution"] = {int(k): int(v) for k, v in df.Star_Rating.value_counts().sort_index().items()}

# ---------- text length ----------
df["Word_Count"] = df.Original_Review_Text.str.split().str.len()
report["word_count"] = {k: float(round(v, 1)) for k, v in df.Word_Count.describe().items()}
report["very_short_reviews_lt_8_words"] = int((df.Word_Count < 8).sum())

# ---------- processed text & freeze ----------
df["Processed_Text"] = df.Original_Review_Text.apply(clean_text)
cols = ["Review_ID", "Attraction_Name", "Attraction_Category", "Original_Review_Text",
        "Processed_Text", "Star_Rating", "Rating_Label", "Final_Sentiment", "Manual_Override",
        "Word_Count", "Language"]
final = df[cols].sort_values("Review_ID").reset_index(drop=True)
path = os.path.join(OUT, "reviews_publication_final.csv")
final.to_csv(path, index=False)
report["frozen_file"] = "data/processed/reviews_publication_final.csv"
report["frozen_rows"] = int(len(final))
report["frozen_sha256"] = hashlib.sha256(open(path, "rb").read()).hexdigest()

per_attr = pd.crosstab([final.Attraction_Category, final.Attraction_Name], final.Final_Sentiment)
per_attr["Total"] = per_attr.sum(axis=1)
per_attr["Mean_Stars"] = final.groupby(["Attraction_Category", "Attraction_Name"]).Star_Rating.mean().round(2)
per_attr.to_csv(os.path.join(AUD, "per_attraction_counts.csv"))

# ---------- locked stratified split ----------
ids, y = final.Review_ID.values, final.Final_Sentiment.values
tr, tmp, ytr, ytmp = train_test_split(ids, y, test_size=0.30, stratify=y, random_state=SEED)
va, te, _, _ = train_test_split(tmp, ytmp, test_size=0.50, stratify=ytmp, random_state=SEED)
split = {"seed": SEED, "scheme": "stratified 70/15/15 on Final_Sentiment",
         "train_ids": sorted(map(int, tr)), "val_ids": sorted(map(int, va)), "test_ids": sorted(map(int, te))}
for k in ["train", "val", "test"]:
    sub = final[final.Review_ID.isin(split[f"{k}_ids"])]
    split[f"{k}_distribution"] = sub.Final_Sentiment.value_counts().to_dict()
json.dump(split, open(os.path.join(OUT, "split_main.json"), "w"), indent=1)
report["split_sizes"] = {k: len(split[f"{k}_ids"]) for k in ["train", "val", "test"]}

json.dump(report, open(os.path.join(AUD, "audit_report.json"), "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in report.items() if k != "override_ids"}, indent=1, ensure_ascii=False))
