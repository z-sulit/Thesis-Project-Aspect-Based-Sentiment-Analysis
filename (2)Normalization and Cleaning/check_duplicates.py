import sys
import pandas as pd
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Path resolution
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "Dataset"

if not DATASET_DIR.exists():
    DATASET_DIR = Path(__file__).resolve().parent / "Dataset"

coffee_file = DATASET_DIR / "coffeeshops_master.csv"
matcha_file = DATASET_DIR / "matcha_shops_master.csv"

dfs = []

# Load from master files if available, otherwise read all individual CSVs
if coffee_file.exists():
    df_c = pd.read_csv(coffee_file)
    df_c["Shop_Type"] = "Coffee"
    dfs.append(df_c)
else:
    c_files = list((DATASET_DIR / "Coffeeshops").glob("*.csv"))
    if c_files:
        df_c = pd.concat([pd.read_csv(f) for f in c_files], ignore_index=True)
        df_c["Shop_Type"] = "Coffee"
        dfs.append(df_c)

if matcha_file.exists():
    df_m = pd.read_csv(matcha_file)
    df_m["Shop_Type"] = "Matcha"
    dfs.append(df_m)
else:
    m_files = list((DATASET_DIR / "Matcha Shops").glob("*.csv"))
    if m_files:
        df_m = pd.concat([pd.read_csv(f) for f in m_files], ignore_index=True)
        df_m["Shop_Type"] = "Matcha"
        dfs.append(df_m)

if not dfs:
    raise FileNotFoundError(f"No CSV files found in {DATASET_DIR}")

df_all = pd.concat(dfs, ignore_index=True)

# Pre-clean review_text to separate actual text reviews vs rating-only (null/empty)
df_all['review_text_str'] = df_all['review_text'].fillna('').astype(str).str.strip()
text_reviews_only = df_all[df_all['review_text_str'] != ''].copy()

# ---------------------------------------------------------
# DUPLICATE CALCULATIONS
# ---------------------------------------------------------
# 1. Duplicates by place_name + review_text (All rows including NaNs)
dup_place_text_all = df_all.duplicated(subset=['place_name', 'review_text']).sum()

# 2. Duplicates by review_text ONLY (Non-empty text reviews)
dup_text_only = text_reviews_only.duplicated(subset=['review_text_str']).sum()

# 3. Duplicates by place_name + review_text (Non-empty text reviews)
dup_place_text_nonempty = text_reviews_only.duplicated(subset=['place_name', 'review_text_str']).sum()

# ---------------------------------------------------------
# SUMMARY TABLES
# ---------------------------------------------------------
summary_data = [
    {
        "Check Criteria": "Place Name + Review Text (Includes Empty NaNs)",
        "Total Rows Analyzed": len(df_all),
        "Duplicate Rows": dup_place_text_all,
        "Unique Rows": len(df_all) - dup_place_text_all,
        "Duplicate %": f"{(dup_place_text_all / len(df_all)) * 100:.2f}%"
    },
    {
        "Check Criteria": "Place Name + Review Text (Non-empty text only)",
        "Total Rows Analyzed": len(text_reviews_only),
        "Duplicate Rows": dup_place_text_nonempty,
        "Unique Rows": len(text_reviews_only) - dup_place_text_nonempty,
        "Duplicate %": f"{(dup_place_text_nonempty / len(text_reviews_only)) * 100:.2f}%"
    },
    {
        "Check Criteria": "Review Text Only (Non-empty text across all shops)",
        "Total Rows Analyzed": len(text_reviews_only),
        "Duplicate Rows": dup_text_only,
        "Unique Rows": len(text_reviews_only) - dup_text_only,
        "Duplicate %": f"{(dup_text_only / len(text_reviews_only)) * 100:.2f}%"
    }
]

summary_df = pd.DataFrame(summary_data)

print("\n" + "=" * 80)
print("REVIEW TEXT DUPLICATE ANALYSIS SUMMARY TABLE")
print("=" * 80)
print(summary_df.to_string(index=False))

# ---------------------------------------------------------
# TOP DUPLICATE REVIEW TEXTS
# ---------------------------------------------------------
print("\n" + "=" * 80)
print("TOP 10 MOST FREQUENTLY DUPLICATED REVIEW TEXTS")
print("=" * 80)

top_dup_texts = (
    text_reviews_only.groupby('review_text_str')
    .agg(
        Occurrences=('place_name', 'count'),
        Unique_Shops=('place_name', 'nunique'),
        Sample_Shop=('place_name', 'first')
    )
    .reset_index()
    .query('Occurrences > 1')
    .sort_values(by='Occurrences', ascending=False)
    .head(10)
)

top_dup_texts.columns = ['Review Text', 'Occurrences', 'Unique Shops Count', 'Sample Shop']
print(top_dup_texts.to_string(index=False))
print("=" * 80 + "\n")