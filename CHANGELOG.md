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
* **Master Dataset Concatenation & Dataset Bifurcation**:
  * Merged **360 individual Coffeeshop CSVs** (17,454 reviews) and **51 individual Matcha Shop CSVs** (2,132 reviews) into `Dataset/coffeeshops_master.csv` and `Dataset/matcha_shops_master.csv`.
  * **Dual Master File Architecture**:
    * `Dataset/DB_master_reviews_combined.csv` (19,586 rows): Raw combined dataset allocated for database storage and dashboard analytics.
    * `Dataset/Thesis_master_reviews_combined.csv` (19,586 rows): Dedicated dataset for AI model training and testing pipelines.
* **Demojization & Emoji Normalization**:
  * Applied `emoji.demojize` across `Thesis_master_reviews_combined.csv`, creating a normalized `clean_review` column.
  * Converts Unicode emojis into text tokens (e.g., `:red_heart:`, `:thumbs_up:`) to enable proper tokenization in transformer architectures (XLM-RoBERTa) while preserving sentiment context.
* **Metadata & Overlap Reporting**:
  * Built `shops_list.txt` enumerating all 411 processed establishments.
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
  * Re-architected `EDA.py` and reporting scripts to stream directly from master CSVs and output `eda_report.pdf`.

---

## 4. Documentation & Paper Alignments

* **Paper Alignment**: Documented data acquisition pipelines and LLM validation steps.
* **Repo Organization**: Cleaned workspace directories and updated README links.

---

## 5. Exploratory Data Analysis and Methodological Justification

### 5.1. Semantic Divergence and Taxonomy Validation
Analysis of bigram frequencies reveals a fundamental divergence in consumer evaluation criteria between the two target markets. Coffee establishment reviews are heavily weighted toward experiential and supplementary factors, dominated by bigrams such as "good food" (462), "coffee shop" (430), and "nice place" (393). In contrast, matcha consumers exhibit a hyper-fixated, product-centric evaluation, highlighted by "matcha latte" (73) and "best matcha" (58).

This semantic split validates the deployment of Generative Aspect Taxonomy Induction. Applying a static, generalized food and beverage taxonomy would fail to capture that coffee shop ratings in this dataset are environmentally driven, while matcha ratings are product-driven.

### 5.2. Lexical Diversity and the Failure of Static Dictionaries
The Type-Token Ratio (TTR) exposes significant disparities in vocabulary richness across the domains. The coffee corpus demonstrates high lexical repetition (TTR = 0.0823), indicating consumers rely on generic, recurring phrasing to describe their visits. Conversely, the matcha corpus exhibits more than double the descriptive diversity (TTR = 0.1718). This variance confirms that a rigid, rule-based keyword extraction approach would artificially truncate the nuanced, highly variable vocabulary utilized by matcha consumers.

### 5.3. Code-Switching Density and Architectural Necessity
The code-switching density analysis provides the exact empirical baseline necessitating the XLM-RoBERTa architecture. The data shows a consistent 5% baseline inclusion of regional Bislish and Taglish conversational markers across both categories (Coffee: 5.05%, Matcha: 4.97%).

While 5% represents a minority of the overall text, these localized terms (e.g., lami, kaayo, mahal) frequently carry the core emotional polarity of the sentence. Processing this corpus with monolingual English lexicons (e.g., VADER) guarantees a baseline failure rate where the most critical sentiment drivers are either misinterpreted or completely discarded as out-of-vocabulary noise. Consequently, deploying a multilingual transformer equipped to resolve intra-sentential code-switching is a structural requirement, not an optional enhancement, for accurately modeling this regional dataset.

---

## 6. Phase 2: Aspect Taxonomy Initialization (`2_AspectTaxonomy/AspectTaxonomy.ipynb`)

* **Initialization of Aspect Taxonomy Pipeline**:
  * Established Phase 2 workflow in `2_AspectTaxonomy/AspectTaxonomy.ipynb` structured around three core stages:
    * **Step 2.1**: Representative Stratified Sampling
    * **Step 2.2**: Generative Semantic Analysis: (Changes)
      * The stratified sample is processed via a Generative Large Language Model. The prompt design instructs the model to act as a qualitative researcher tasked with inducing a primary aspect taxonomy representing critical dimensions of consumer feedback. While the model must comprehend the regional Davao City consumer context and Bislish semantics to accurately map localized sentiments, it is strictly constrained to output broad semantic aspect categories (e.g., Ambiance, Customer Service) in a standardized JSON format. This strict formatting enables the programmatic execution of the zero-shot self-correction loop.
    * **Step 2.3**: Global Taxonomy Formalization and Validation
* **Dataset Ingestion & Dialect Lexicon Setup**:
  * Ingested processed review data from `Dataset/Thesis_master_reviews_combined.csv`.
  * Curated localized lexical marker dictionaries encompassing Bisaya/Cebuano, Tagalog, and shared regional terms to support dialect-aware aspect extraction and semantic parsing.

---

## 7. Generative AI Methodological Justifications & Solution Verification

### 7.1. Phase 2: Generative Aspect Taxonomy Induction


* **Zero-Shot Self-Correction**:
  * Grounded in the methodology proposed by Brady & T. Islam.
  * Employs prompt-based inference to iteratively refine and shape the aspect taxonomy using Large Language Models without requiring manual seed sets.
  * Enables the AI to logically organize and structure semantic aspect labels.

  self-refine: https://arxiv.org/pdf/2303.17651


### 7.2. Phase 5: Targeted Synthetic Data Augmentation

Cosine-similarity: https://arxiv.org/pdf/1908.10084
* **Cosine Similarity (Semantic Overlap)**:
  * The Mechanic: It converts sentences into dense vector embeddings (using a lightweight model like Sentence-BERT) and measures the angle between those vectors in a multidimensional space.
