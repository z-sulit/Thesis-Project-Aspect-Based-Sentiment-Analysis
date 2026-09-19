### [📖 README](README.md) &nbsp;&nbsp;|&nbsp;&nbsp; [🔔 CHANGELOG](CHANGELOG.md)

## How to Clone the Repository
---

```bash
git clone https://github.com/z-sulit/Thesis-Project-Aspect-Based-Sentiment-Analysis.git
cd Thesis-Project-Aspect-Based-Sentiment-Analysis
```

## Git Workflow & Basic Commands
---

### 1. Pulling the Latest Changes
Before starting work, update your local repository with the latest changes from the remote repository:
```bash
git pull origin main
```

### 2. Checking Working Status
Check which files have been modified, deleted, or untracked:
```bash
git status
```

### 3. Staging and Committing Changes
Stage specific files or all modified files, then commit with a descriptive message:
```bash
# Stage a specific file
git add <file_path>

# Or stage all changed files
git add .

# Commit staged changes
git commit -m "Brief summary of changes"
```

### 4. Pushing Changes to Remote
Push your local commits to the GitHub repository:
```bash
git push origin main
```

### 5. Working with Branches
Create and manage feature branches for isolated development:
```bash
# Create and switch to a new branch
git checkout -b feature-name

# Push a new branch to remote
git push -u origin feature-name

# Switch back to the main branch
git checkout main
```

## Setting Up the Virtual Environment
---

### Why Use a Virtual Environment?
* **Dependency Isolation:** Prevents package version conflicts between different projects.
* **System Cleanliness:** Keeps the system-wide Python installation clean and avoids requiring administrative permissions.
* **Reproducibility:** Ensures the exact dependencies can be easily exported (e.g., via `requirements.txt`) and recreated.
* **Multi-Version Testing:** Allows testing the application under different Python or package versions without affecting other environments.

Follow these steps to set up a Python virtual environment to run scripts and Jupyter notebooks with the correct dependencies:

### 1. Create the Virtual Environment
Run the following command in your terminal from the project root directory:
```bash
python -m venv .venv
```

### 2. Activate the Virtual Environment
* **Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
* **Windows (Command Prompt):**
  ```cmd
  .venv\Scripts\activate.bat
  ```
* **macOS / Linux:**
  ```bash
  source .venv/bin/activate
  ```

### 3. Install Dependencies
Ensure you have the virtual environment activated (you should see `(.venv)` in your terminal prompt) and install the required packages:
```bash
pip install --upgrade pip
pip install pandas notebook ipykernel
# Add other packages here if needed, or:
# pip install -r requirements.txt
```

### 4. Link the Virtual Environment to Jupyter Notebook
To ensure your Jupyter notebooks use the virtual environment's packages, register it as a Jupyter kernel:
```bash
python -m ipykernel install --user --name=.venv --display-name "Python (.venv)"
```

When you open a Jupyter Notebook, make sure to select the **Python (.venv)** kernel from the top-right corner of the interface (or under **Kernel -> Change Kernel**).

* **VS Code Users:** If you are using VS Code, click on **Select Kernel** in the upper right corner of the notebook and choose the `.venv` Python interpreter/kernel.

# Aspect-Based Sentiment Analysis of Code-Switched Consumer Reviews

## Introduction
This repository outlines an Aspect-Based Sentiment Analysis (ABSA) system designed to process unstructured, multilingual, and code-switched consumer reviews of local coffee and matcha businesses in Davao City. As detailed in the foundational research document, **Faberes_Mamac_Sulit_DS-Paper.pdf**, traditional lexicon-based sentiment analysis tools fail when processing localized dialects like Taglish and Bislish. To solve this, the project deploys a fine-tuned XLM-ROBERTa model, integrated with Active Learning and regulated Synthetic Data Generation, to extract actionable market intelligence from noisy textual data.

## Objectives
* **Category Induction:** Discover and categorize operational aspects (e.g., taste, ambiance, parking) specific to the Davao City cafe market.
* **Model Benchmarking:** Demonstrate the superior performance of multilingual transformers (XLM-ROBERTa) over lexicon-based approaches (e.g., VADER) on code-switched inputs.
* **Annotation Efficiency:** Utilize Active Learning (uncertainty and diversity sampling) to drastically reduce manual labeling costs.
* **Data Scarcity Mitigation:** Apply LLM-based Self-Instruct frameworks to synthesize training data for underrepresented classes without causing model collapse.

## System Pipeline & Empirical Results

The architecture follows a rigorous multi-stage pipeline:

1. **Data Acquisition & Preprocessing:** Cleaned and demojized 19,586 Google Maps consumer reviews across 411 Davao City establishments ([`(2)Normalization and Cleaning/Data preprocess.ipynb`](file:///D:/Ateneo%20de%20Davao/Thesis-Project-Aspect-Based-Sentiment-Analysis/(2)Normalization%20and%20Cleaning/Data%20preprocess.ipynb)).
2. **Aspect Taxonomy Induction:** Generative zero-shot self-correction loop establishing localized aspect categories ([`2_AspectTaxonomy/AspectTaxonomy.ipynb`](file:///D:/Ateneo%20de%20Davao/Thesis-Project-Aspect-Based-Sentiment-Analysis/2_AspectTaxonomy/AspectTaxonomy.ipynb)).
3. **Aspect Extraction & Data Decomposition:** Multi-aspect sentence explosion into isolated review-aspect pairs ([`_3_Granular ASpect Extraction and Data Decomposition/gaedd.ipynb`](file:///D:/Ateneo%20de%20Davao/Thesis-Project-Aspect-Based-Sentiment-Analysis/_3_Granular%20ASpect%20Extraction%20and%20Data%20Decomposition/gaedd.ipynb)).
4. **Strict Data Partitioning:** 70% Train, 15% Validation, and 15% Test stratified partition to enforce the evaluation firewall ([`_4_Strict_Data_Parititioning/DataPartition.ipynb`](file:///D:/Ateneo%20de%20Davao/Thesis-Project-Aspect-Based-Sentiment-Analysis/_4_Strict_Data_Parititioning/DataPartition.ipynb)).
5. **Targeted Synthetic Data Augmentation (Phase 5):** Evaluated learning curve sufficiency plateaus and addressed extreme negative sentiment sparsity across minority aspect classes. Developed an LLM-assisted generation pipeline with dual-stage deduplication using TF-IDF and Cosine Similarity (threshold 0.85) against authentic data, generating filtered negative candidates flagged in `Train_Set_70_Augmented.csv` ([`_5_Targeted_Synthetic_Data_Augmentation/Data_augmentation.ipynb`](file:///D:/Ateneo%20de%20Davao/Thesis-Project-Aspect-Based-Sentiment-Analysis/_5_Targeted_Synthetic_Data_Augmentation/Data_augmentation.ipynb)).
6. **Active Learning with Human-in-the-Loop & Regulated Augmentation (Phase 6):** Iterative uncertainty sampling, multi-annotator agreement, and controlled dataset expansion ([`_6_Active_Learning_w_Human_Annotation/phase6.ipynb`](file:///D:/Ateneo%20de%20Davao/Thesis-Project-Aspect-Based-Sentiment-Analysis/_6_Active_Learning_w_Human_Annotation/phase6.ipynb)):
   * **Seed Pool & Baseline Engine:** Formulated a 100-row stratified seed pool (`Seed_Ground_Truth.csv`) across 1–5 star ratings and all aspects, validated across 4 independent annotators (Fleiss' Kappa $\kappa = 0.6550$) with majority voting and tie-breaking. Fine-tuned the initial `xlm-roberta-base` baseline using native PyTorch.
   * **Batch 2 Uncertainty Sampling:** Executed margin-based inference ($Top_1 - Top_2$ probability difference) across unannotated candidates (`Unannotated_Pool_Ranked_Margin.csv`), extracting and annotating 400 high-uncertainty instances to reach 500 rows (`Cumulative_Ground_Truth_500.csv`).
   * **Controlled 30% Synthetic Injection:** Injected 430 deduplicated synthetic negative rows with redistributed aspect quotas (113 Ambiance, 93 Facilities, 112 Price, 112 Store Operations) to establish a 1,434-row balanced master training dataset (`Augmented_Ground_Truth_1434.csv`), strictly enforcing a 29.98% synthetic ceiling to prevent model collapse and preserve authentic Bislish/Taglish semantics.
   * **Inter-Rater Reliability & Verification:** Implemented automated audit tooling (`eda_phase6.py`) computing Fleiss' Kappa per annotation batch, generating aspect-by-sentiment cross-tabulations, and outputting distribution reports.

## Researchers
* Audrey Zarina Faberes
* Ira Zaky O. Mamac
* Zachary Lorenzo F. Sulit
    * From *Ateneo de Davao University*

## Dataset Schema

The processed dataset is aggregated across 411 establishments (360 coffee shops, 51 matcha shops) totaling 19,586 reviews, stored in [`Dataset/Thesis_master_reviews_combined.csv`](file:///D:/Ateneo%20de%20Davao/Thesis-Project-Aspect-Based-Sentiment-Analysis/Dataset/Thesis_master_reviews_combined.csv).

| Column Field | Data Type | Description |
| :--- | :--- | :--- |
| `place_name` | `string` | Name of the target coffee or matcha establishment |
| `latitude` | `float` | Geographic latitude of the business location |
| `longitude` | `float` | Geographic longitude of the business location |
| `rating` | `integer` | Numerical rating given by the user (1 to 5 stars) |
| `review_text` | `string` | Raw review text extracted from Google Maps |
| `clean_review` | `string` | Demojized text with normalized whitespace and standardized token markers |
| `Shop_Type` | `string` | Establishment category classification (`Coffee` or `Matcha`) |

## Evaluation Metrics & Benchmarks

* **Performance Metrics:** Macro F1-Score, Precision, Recall, and Accuracy evaluated across both aspect detection and sentiment polarity classification (Positive, Negative, Neutral).
* **Inter-Annotator Agreement:** Validated via Fleiss' Kappa ($\kappa$) for human-in-the-loop Active Learning annotations.
* **Baseline Benchmarks:** Comparative evaluation of fine-tuned **XLM-RoBERTa** against lexicon-based baselines (**VADER**) on localized code-switched (Bislish/Taglish) inputs.

## Citation

If you use this dataset or research pipeline, please cite:

```bibtex
@article{faberes2026aspect,
  title={Aspect-Based Sentiment Analysis of Code-Switched Consumer Reviews in Davao City},
  author={Faberes, Audrey Zarina and Mamac, Ira Zaky O. and Sulit, Zachary Lorenzo F.},
  journal={Department of Computer Science, Ateneo de Davao University},
  year={2026}
}
```

