## How to Clone the Repository

```bash
git clone https://github.com/z-sulit/Thesis-Project-Aspect-Based-Sentiment-Analysis.git
cd Thesis-Project-Aspect-Based-Sentiment-Analysis
```

# Aspect-Based Sentiment Analysis of Code-Switched Consumer Reviews

## Introduction
This repository outlines an Aspect-Based Sentiment Analysis (ABSA) system designed to process unstructured, multilingual, and code-switched consumer reviews of local coffee and matcha businesses in Davao City. As detailed in the foundational research document, **Faberes_Mamac_Sulit_DS-Paper.pdf**, traditional lexicon-based sentiment analysis tools fail when processing localized dialects like Taglish and Bislish. To solve this, the project deploys a fine-tuned XLM-ROBERTa model, integrated with Active Learning and regulated Synthetic Data Generation, to extract actionable market intelligence from noisy textual data.

## Objectives
* **Category Induction:** Discover and categorize operational aspects (e.g., taste, ambiance, parking) specific to the Davao City cafe market.
* **Model Benchmarking:** Demonstrate the superior performance of multilingual transformers (XLM-ROBERTa) over lexicon-based approaches (e.g., VADER) on code-switched inputs.
* **Annotation Efficiency:** Utilize Active Learning (uncertainty and diversity sampling) to drastically reduce manual labeling costs.
* **Data Scarcity Mitigation:** Apply LLM-based Self-Instruct frameworks to synthesize training data for underrepresented classes without causing model collapse.

## System Pipeline
The architecture relies on a multi-stage process:
1. **Data Acquisition:** Bypassing API limits via custom semi-automated DOM scraping on Google Maps.
2. **Taxonomy Induction:** Using generative LLMs (Gemini 3) to organically discover a localized aspect taxonomy rather than relying on static industry dictionaries.
3. **Aspect Extraction:** Decomposing complex reviews into isolated review-aspect pairs to resolve mixed sentiments in single sentences.
4. **Active Learning & Annotation:** Routing strictly ambiguous, high-value samples to human annotators, validated via Fleiss' Kappa.
5. **Targeted Augmentation:** Regulated generation of artificial text to balance minority classes.
6. **Transformer Fine-Tuning:** Training XLM-ROBERTa on the hybrid dataset for final aspect-level sentiment classification.

## Researchers
* Audrey Zarina Faberes
* Ira Zaky O. Mamac
* Zachary Lorenzo F. Sulit
* *Ateneo de Davao University*

## For change in paper:

Data Acquisition changes in paper.

## Implementation Updates:
Ongoing data extraction.
