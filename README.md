### [📖 README](README.md) &nbsp;&nbsp;|&nbsp;&nbsp; [🔔 CHANGELOG](CHANGELOG.md)

# Aspect-Based Sentiment Analysis of Code-Switched Consumer Reviews

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5%2B%20(CUDA%2012.1)-ee4c2c.svg)](https://pytorch.org/)
[![Transformers 4.45+](https://img.shields.io/badge/Transformers-HuggingFace-orange.svg)](https://huggingface.co/)
[![License: Academic](https://img.shields.io/badge/License-Academic%20Use-green.svg)](#citation)

---

## Executive Summary & Research Overview

This repository houses the end-to-end Aspect-Based Sentiment Analysis (ABSA) system designed for unstructured, multi-dialectal, and code-switched consumer reviews of coffee and matcha establishments across Davao City, Philippines. Standard natural language processing tools and monolingual sentiment lexicons fail when parsing localized regional discourse containing Taglish (Tagalog-English) and Bislish (Bisaya/Cebuano-English). 

To solve this, the pipeline integrates **Active Learning with Human-in-the-Loop validation**, **Controlled Synthetic Data Augmentation (capped at 30%)**, and a **fine-tuned XLM-RoBERTa transformer** constrained by structural layer-freezing, dynamic validation checkpointing, and inverse class-weighted Cross-Entropy loss.

```
========================================================================================================
[19,586 Google Maps Reviews] ──> [Normalization & Demojization] ──> [Aspect Taxonomy Induction]
                                                                              │
[Evaluation Firewall: 400 Test / 400 Val] <── [Stratified Partition] <───────┘
                     │
[Active Learning + 30% Synthetic Injection] ──> [714-Row Master Training Split]
                                                              │
                                            [XLM-RoBERTa Grid Search Fine-Tuning]
                                            (Freezing L0-L5, Balanced Inverse Loss)
                                                              │
                                            [Phase 8 Out-of-Sample Benchmark]
                                            (vs. VADER & Tuned SVM TF-IDF + McNemar)
========================================================================================================
```

---

## Repository Architecture & Directory Map

```text
├── (2)Normalization and Cleaning/
│   ├── Data preprocess.ipynb                    # Demojization, normalization, and deduplication
│   └── tt1.ipynb                                # Text cleaning verification scratchpad
├── 2_AspectTaxonomy/
│   └── AspectTaxonomy.ipynb                     # Generative zero-shot aspect category induction
├── Dataset/
│   ├── Coffeeshops/                             # 360 raw scraped coffee establishment CSVs
│   ├── Matcha Shops/                            # 51 raw scraped matcha establishment CSVs
│   ├── DB_master_reviews_combined.csv           # 19,586 raw reviews for analytics storage
│   └── Thesis_master_reviews_combined.csv       # 19,586 cleaned & demojized reviews for AI modeling
├── Thesis-datacollectionscripts/
│   └── DOMscript.js                             # Client-side automated scraper for Google Maps
├── _3_Granular ASpect Extraction and Data Decomposition/
│   └── gaedd.ipynb                              # Review explosion into review-aspect pairs
├── _4_Strict_Data_Parititioning/
│   ├── DataPartition.ipynb                      # Stratified split (70% Train, 15% Val, 15% Test)
│   ├── eval_sampling.ipynb                      # Sampling verification
│   └── test.ipynb                               # Data integrity checks
├── _5_Targeted_Synthetic_Data_Augmentation/
│   ├── Data_augmentation.ipynb                 # LLM generation with cosine similarity filtering
│   └── _5_eda.ipynb                             # Augmentation distribution analysis
├── _6_Active_Learning_w_Human_Annotation/
│   ├── phase6.ipynb                             # Margin-based uncertainty sampling & Fleiss' Kappa
│   └── phase6_nt2.ipynb                         # Active learning round 2 iterations
├── _7_Model Fine-Tuning/
│   ├── Augmented_Ground_Truth_714.csv           # 714-row training set (500 authentic + 214 synthetic)
│   ├── validation_set_400_final.csv             # 400-row authentic human-annotated validation set
│   ├── test_set_400_final.csv                   # 400-row authentic out-of-sample test firewall
│   ├── phase7_weighted_loss.ipynb               # Inverse class-weighted grid search & evaluation
│   ├── phase7_new.ipynb                         # Baseline smoothed training and Phase 8 suite
│   ├── xlm_roberta_calibrated_phase7/           # Calibrated model checkpoint (smoothed baseline)
│   ├── xlmr_absa_weighted/                      # Calibrated model checkpoint (weighted loss)
│   ├── Final_Benchmark_Results.csv              # Benchmark metrics (Accuracy, Macro-F1, Class F1s)
│   ├── McNemar_Test_Results.csv                 # McNemar pairwise contingency tests
│   ├── Qualitative_Error_Analysis_Cases.csv     # Model winning edge cases
│   ├── XLM_RoBERTa_Failure_Cases.csv            # Error cases for qualitative discussion
│   ├── Phase7_Final_Results.pdf                 # Training curves & validation classification report
│   └── Phase8_Final_Results.pdf                 # Confusion matrices & benchmark visual summary
├── CHANGELOG.md                                 # Chronological technical implementation logs
└── README.md                                    # Project documentation and reproduction guide
```

---

## End-to-End System Pipeline

### Phase 1 & 2: Acquisition, Normalization & Linguistic Profiling
* **Target & Scope:** Ingested 19,586 customer reviews across 411 Davao City establishments (360 coffee shops, 51 matcha shops) using [`DOMscript.js`](file:///D:/Ateneo%20de%20Davao/Thesis-Project-Aspect-Based-Sentiment-Analysis/Thesis-datacollectionscripts/DOMscript.js).
* **Demojization & Normalization:** Applied `emoji.demojize` to standardize emojis into readable text tokens (e.g., `:red_heart:`, `:thumbs_up:`) to preserve non-verbal sentiment markers without token vocabulary fragmentation in [`Data preprocess.ipynb`](file:///D:/Ateneo%20de%20Davao/Thesis-Project-Aspect-Based-Sentiment-Analysis/(2)Normalization%20and%20Cleaning/Data%20preprocess.ipynb).
* **Code-Switching Verification:** Identified a consistent **3.20% to 5.05%** lexical density of Cebuano/Bisaya and Tagalog polarity anchors (`lami`, `kaayo`, `way lami`, `mahal`, `barato`), empirically confirming the invalidity of English-only lexicons.

### Phase 3: Generative Aspect Taxonomy Induction
* Executed an LLM-guided zero-shot aspect discovery and correction loop on stratified samples in [`AspectTaxonomy.ipynb`](file:///D:/Ateneo%20de%20Davao/Thesis-Project-Aspect-Based-Sentiment-Analysis/2_AspectTaxonomy/AspectTaxonomy.ipynb).
* Formalized core localized evaluation dimensions: `Product Quality`, `Customer Service`, `Ambiance`, `Facilities and Amenities`, `Price and Value`, and `Store Operations`.

### Phase 4: Granular Aspect Extraction & Decomposition
* Transformed complex multi-aspect sentences into isolated `(TargetAspect, sentence)` dyads in [`gaedd.ipynb`](file:///D:/Ateneo%20de%20Davao/Thesis-Project-Aspect-Based-Sentiment-Analysis/_3_Granular%20ASpect%20Extraction%20and%20Data%20Decomposition/gaedd.ipynb), ensuring atomic, independent sentiment labeling per review statement.

### Phase 5: Strict Partitioning & Evaluation Firewall
* Structured a strict 70% Train, 15% Validation, and 15% Test partition in [`DataPartition.ipynb`](file:///D:/Ateneo%20de%20Davao/Thesis-Project-Aspect-Based-Sentiment-Analysis/_4_Strict_Data_Parititioning/DataPartition.ipynb).
* Enforced an uncompromised evaluation firewall: all test and validation partitions consist exclusively of **100% authentic, human-annotated** reviews (`validation_set_400_final.csv` and `test_set_400_final.csv`) with zero synthetic contamination.

### Phase 6: Active Learning & Controlled Synthetic Augmentation
* **Active Learning Loop:** Deployed margin-based uncertainty sampling ($Top_1 - Top_2$ probability delta) to prioritize ambiguous reviews for multi-annotator labeling in [`phase6.ipynb`](file:///D:/Ateneo%20de%20Davao/Thesis-Project-Aspect-Based-Sentiment-Analysis/_6_Active_Learning_w_Human_Annotation/phase6.ipynb), achieving substantial inter-rater reliability (Fleiss' Kappa $\kappa = 0.6550$).
* **Regulated Augmentation:** Synthesized targeted minority-aspect negative cases via Self-Instruct prompts filtered by TF-IDF and Cosine Similarity thresholds (0.85 ceiling). Maintained a strict **29.98% synthetic ceiling** (500 authentic + 214 synthetic = 714 training rows in `Augmented_Ground_Truth_714.csv`) to insulate the transformer against representation collapse.

### Phase 7: Model Fine-Tuning & Hyperparameter Grid Search
* **Model Selection:** `xlm-roberta-base` (270M parameters) deployed for its 250k-token SentencePiece shared vocabulary and pre-training across low-resource Philippine languages (Tagalog, Cebuano).
* **Structural Safeguards:**
  * **Layer Freezing:** Token embeddings and bottom 6 encoder layers (Layers 0 to 5) strictly frozen, cutting trainable parameters down to ~43.1M to eliminate catastrophic memorization on small corpora.
  * **Regularization:** AdamW with decoupled weight decay of 0.1 and dropout rate of 0.1 on all active layers.
  * **Warmup:** 10% linear warmup schedule to absorb initial gradient shocks.
* **Loss Objectives:**
  1. *Baseline Formulation:* Label-Smoothed Cross-Entropy (`label_smoothing=0.1`) in [`phase7_new.ipynb`](file:///D:/Ateneo%20de%20Davao/Thesis-Project-Aspect-Based-Sentiment-Analysis/_7_Model%20Fine-Tuning/phase7_new.ipynb).
  2. *Cost-Sensitive Formulation:* Balanced Inverse Class-Weighted Cross-Entropy (`label_smoothing=0.0`) in [`phase7_weighted_loss.ipynb`](file:///D:/Ateneo%20de%20Davao/Thesis-Project-Aspect-Based-Sentiment-Analysis/_7_Model%20Fine-Tuning/phase7_weighted_loss.ipynb), applying balanced class weights:
     $$w_c = \frac{N}{K \cdot N_c} \implies w_{\text{Positive}} \approx 0.58,\quad w_{\text{Negative}} \approx 0.89,\quad w_{\text{Neutral}} \approx 6.10$$
* **Grid Search:** Tested batch sizes $\{16, 32\}$ and learning rates $\{2\times 10^{-5}, 3\times 10^{-5}, 5\times 10^{-5}\}$ over 5 epochs; optimal checkpoint selected strictly by peak Validation Macro-F1.

### Phase 8: Out-of-Sample Performance Evaluation & Validation
* **Comparative Baselines:**
  * **VADER Lexicon:** Unconditioned lexical rule benchmark using standard polarity boundaries ($\ge 0.05$ Positive, $\le -0.05$ Negative, else Neutral).
  * **Aspect-Conditioned SVM (TF-IDF):** 5-fold cross-validated grid search tuning $C \in \{0.1, 1.0, 10.0\}$, n-gram ranges $\{(1,1), (1,2)\}$, max features $\{3000, 5000\}$, and class weighting.
* **Statistical Rigor:** McNemar's non-parametric test evaluated on discordant prediction pairs ($b$ vs. $c$) on the identical 400-row test split:
  $$\chi^2 = \frac{(b - c)^2}{b + c},\quad \alpha = 0.05$$
* **Error Analysis:** Categorization of winning edge cases (`Qualitative_Error_Analysis_Cases_Weighted.csv`) and failure modes (`XLM_RoBERTa_Failure_Cases_Weighted.csv`) to assess contextual linguistic nuances.

---

## Empirical Benchmark Results (Test Set $N=400$)

| Model | Accuracy | Macro Precision | Macro Recall | **Macro-F1** | F1 (POS) | F1 (NEU) | F1 (NEG) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **VADER Lexicon** | 0.6875 | 0.5685 | 0.5052 | **0.4985** | 0.8293 | 0.2034 | 0.4630 |
| **Aspect-Conditioned SVM (TF-IDF)** | 0.7300 | 0.5123 | 0.5290 | **0.5187** | 0.8502 | 0.1875 | 0.5185 |
| **XLM-RoBERTa (Smoothed Baseline)** | **0.8450** | **0.5131** | **0.6063** | **0.5508** | **0.9302** | 0.0000 | **0.7222** |
| **XLM-RoBERTa (Inverse Class-Weighted)** | *Calibrated* | *Evaluated* | *Evaluated* | *Phase 7/8* | *Reported* | *Penalized* | *Reported* |

### McNemar's Statistical Significance ($\alpha = 0.05$)

| Baseline Comparison | XLM-R Correct ($b$) | Baseline Correct ($c$) | $\chi^2$ Statistic | $p$-value | Significance |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XLM-RoBERTa vs. VADER** | 85 | 22 | **37.0935** | $1.126\times 10^{-9}$ | **Statistically Significant ($p < 0.001$)** |
| **XLM-RoBERTa vs. Tuned SVM** | 58 | 12 | **30.2286** | $3.840\times 10^{-8}$ | **Statistically Significant ($p < 0.001$)** |

---

## Installation & Environment Setup

### 1. Clone Repository
```bash
git clone https://github.com/z-sulit/Thesis-Project-Aspect-Based-Sentiment-Analysis.git
cd Thesis-Project-Aspect-Based-Sentiment-Analysis
```

### 2. Create and Activate Virtual Environment
```bash
# Create venv
python -m venv .venv

# Activate on Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Activate on Linux / macOS
source .venv/bin/activate
```

### 3. Install GPU-Accelerated PyTorch & Dependencies
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch with CUDA 12.1 runtime
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core scientific and NLP libraries
pip install pandas numpy scikit-learn transformers statsmodels nltk seaborn matplotlib ipykernel ipywidgets
```

### 4. Register Jupyter Kernel
```bash
python -m ipykernel install --user --name=.venv --display-name "Python (.venv)"
```

---

## Dataset Schema

The processed master dataset [`Dataset/Thesis_master_reviews_combined.csv`](file:///D:/Ateneo%20de%20Davao/Thesis-Project-Aspect-Based-Sentiment-Analysis/Dataset/Thesis_master_reviews_combined.csv) structures 19,586 consumer reviews across 411 establishments:

| Column Field | Data Type | Description |
| :--- | :--- | :--- |
| `place_name` | `string` | Registered establishment business name |
| `latitude` | `float` | Geographic latitude coordinate |
| `longitude` | `float` | Geographic longitude coordinate |
| `rating` | `integer` | Consumer star rating ($1$ to $5$) |
| `review_text` | `string` | Raw review text scraped from Google Maps |
| `clean_review` | `string` | Demojized and normalized text |
| `Shop_Type` | `string` | Business classification (`Coffee` or `Matcha`) |

---

## Research Team & Affiliation

* **Audrey Zarina Faberes**
* **Ira Zaky O. Mamac**
* **Zachary Lorenzo F. Sulit**

*Department of Computer Science, School of Computing*  
**Ateneo de Davao University**, Davao City, Philippines

---

## Citation

```bibtex
@article{faberes2026aspect,
  title={Aspect-Based Sentiment Analysis of Code-Switched Consumer Reviews in Davao City},
  author={Faberes, Audrey Zarina and Mamac, Ira Zaky O. and Sulit, Zachary Lorenzo F.},
  journal={Department of Computer Science, Ateneo de Davao University},
  year={2026}
}
```
