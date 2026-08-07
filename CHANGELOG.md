### [📖 README](README.md) &nbsp;&nbsp;|&nbsp;&nbsp; [🔔 CHANGELOG](CHANGELOG.md)

# Changelog

All notable changes, implementation updates, data processing steps, and exploratory data analysis (EDA) updates for the Aspect-Based Sentiment Analysis (ABSA) project are documented in this file.

---

## 1. Data Collection & DOM Script Architecture (`DOMscript.js`)

* **Target & Scope**: Automated client-side extraction script tailored for Google Maps review pages.
* **Extraction Payload**: Captures establishment `place_name`, geographic coordinates (`latitude`, `longitude`), `rating` (1–5 stars), and cleaned `review_text`.
* **Owner Response Filtering**: Implemented context checking (`isOwnerReplyText`) to exclude business owner replies from customer review data.
* **Text Normalization & Deduplication**:
  * Automatically clicks "More" / "See more" expansion buttons for truncated text.
  * Sanitizes control characters, non-printable Unicode ranges, and standardizes whitespace.
  * Uses review text fingerprints to prevent duplicate row exports.
* **Export Format**: Exports sanitized CSV files with UTF-8 BOM encoding per business location into designated dataset subdirectories (`Dataset/Coffeeshops` and `Dataset/Matcha Shops`).

---

## 2. Data Preprocessing & Aggregation (`Data preprocess.ipynb`)

* **Category Labeling**: Programmatically appended `Shop_Type` tags (`Coffee` vs `Matcha`) to all extracted establishment files.
* **Master Dataset Concatenation**:
  * Merged **360 individual Coffeeshop CSVs** into `Dataset/coffeeshops_master.csv` (18,258 rows).
  * Merged **51 individual Matcha Shop CSVs** into `Dataset/matcha_shops_master.csv` (2,132 rows).
* **Metadata & Overlap Reporting**:
  * Built `shops_list.txt` enumerating all processed establishments.
  * Generated `duplicate_report.txt` auditing file matches and cross-directory review overlaps (e.g., `Hachi_House_Davao_reviews.csv`).

---

## 3. Exploratory Data Analysis & Linguistic Profiling (`EDA.py` & `generate_dashboard.py`)

* **Establishment Volume Balance**:
  * Analyzed review distribution across 411 unique establishment files.
  * Checked chain dominance: Top 3 establishments account for **7.18%** and Top 10 account for **19.60%** of total dataset volume (Starbucks, Paramount Coffee, and The Coffee Bar leading).
* **Rating Distribution Analysis**:
  * Evaluated 1–5 star rating proportions per category.
  * Calculated 1-to-3 star low rating ratio to establish baseline negative sentiment: **14.30%** for Coffee vs **6.24%** for Matcha.
* **Textual & Linguistic Profiling**:
  * **Length Metrics**: Calculated word count and character count distribution (Mean: 30.8 words, 95th percentile: 108 words, Max: 619 words).
  * **Emoji Frequency**: Extracted and ranked top emojis (Leading: ❤️ [445], 👍 [271], 😊 [175], 😍 [170], ☕ [164]) to quantify non-verbal sentiment indicators.
  * **Code-Switching / Dialect Density**: Detected local Bisaya/Tagalog lexical markers (`lami`, `kaayo`, `sarap`, `pud`, `jud`, `grabe`), revealing a **3.20%** code-switched density (Coffee: 3.27%, Matcha: 2.66%), confirming the necessity for multilingual fine-tuning (XLM-RoBERTa).
* **Vocabulary & N-Gram Analysis**:
  * Extracted Top Unigrams and Bigrams for Coffee ("good food", "coffee shop", "nice place") vs Matcha ("matcha latte", "best matcha", "love matcha").
  * Computed Vocabulary Richness (Type-Token Ratio - TTR): Overall TTR = **0.0327** (Coffee: **0.0342**, Matcha: **0.1030**).
* **Automated PDF Report Generation**:
  * Re-architected `generate_dashboard.py` and `EDA.py` to stream directly from master CSVs.
  * Compiled statistical tables and Matplotlib visualization charts into `dashboard_report.pdf`.

---

## 4. Documentation & Paper Alignments

* **Paper Alignment**: Documented data acquisition pipelines and LLM validation steps.
* **Repo Organization**: Cleaned workspace directories and updated README links.
