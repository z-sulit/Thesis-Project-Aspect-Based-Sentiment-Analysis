import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
import os

def compute_fleiss_kappa(file_paths, sentiment_col='sentiment'):
    """Reads 4 annotator files, extracts ratings, and computes Fleiss' Kappa."""
    ratings = []
    for path in file_paths:
        df = pd.read_csv(path)
        # Clean and standardize annotations
        sent = df[sentiment_col].astype(str).str.strip().str.upper()
        ratings.append(sent)
    
    # Transpose so rows are items, columns are raters
    df_ratings = pd.DataFrame(ratings).T
    
    # Map to integers for statsmodels
    label_map = {'POSITIVE': 0, 'NEGATIVE': 1, 'NEUTRAL': 2}
    df_num = df_ratings.replace(label_map)
    
    # Ensure no unmapped strings exist (like 'NAN' or empty)
    # If any exist, drop those specific rows or handle them. Here we enforce numeric.
    df_num = df_num.apply(pd.to_numeric, errors='coerce').fillna(2).astype(int)
    
    # Aggregate rater counts per item
    agg_ratings, _ = aggregate_raters(df_num.values)
    
    # Calculate Kappa
    kappa = fleiss_kappa(agg_ratings, method='fleiss')
    return kappa

def main():
    print("--- 1. Inter-Rater Reliability (Fleiss' Kappa) ---")
    
    # Seed Batch Files
    seed_files = [
        'Copy 1 Seed_Batch.csv',
        'Copy 2 Seed_Batch.csv',
        'Copy 3 Seed_Batch.csv',
        'Copy 4 Seed_Batch.csv'
    ]
    
    # AL Batch 2 Files
    al2_files = [
        'AL1_Batch_2_For_Annotators.csv',
        'AL2_Batch_2_For_Annotators.csv',
        'AL3_Batch_2_For_Annotators.csv',
        'AL4_Batch_2_For_Annotators.csv'
    ]
    
    try:
        kappa_seed = compute_fleiss_kappa(seed_files)
        print(f"Seed Batch Fleiss' Kappa: {kappa_seed:.4f}")
    except FileNotFoundError:
        print("Seed Batch files not found in the current directory.")
        
    try:
        kappa_al2 = compute_fleiss_kappa(al2_files)
        print(f"AL Batch 2 Fleiss' Kappa: {kappa_al2:.4f}")
    except FileNotFoundError:
        print("AL Batch 2 files not found in the current directory.")

    print("\n--- 2. Ground Truth Distribution (500 Rows) ---")
    # Change to Cumulative_Ground_Truth_1000.csv if you want the 1,000-row version
    #gt_file = 'Cumulative_Ground_Truth_500.csv'
    gt_file = 'v2_Ground_Truth_500.csv'
    
    if not os.path.exists(gt_file):
        print(f"Error: {gt_file} not found.")
        return

    df_gt = pd.read_csv(gt_file)
    
    # Cross-Tabulation
    crosstab = pd.crosstab(df_gt['TargetAspect'], df_gt['sentiment'], margins=True, margins_name="Total")
    print(f"\nCross-Tabulation: TargetAspect by Sentiment ({len(df_gt)} rows):")
    print("-" * 60)
    print(crosstab)
    print("-" * 60)

    # 3. Visualization
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Plot Grouped Bar Chart
    ax = sns.countplot(
        data=df_gt, 
        y='TargetAspect', 
        hue='sentiment', 
        hue_order=['POSITIVE', 'NEGATIVE', 'NEUTRAL'],
        palette={'POSITIVE': '#2ca02c', 'NEGATIVE': '#d62728', 'NEUTRAL': '#7f7f7f'}
    )
    
    plt.title(f'Final Training Data Distribution: Aspect by Sentiment (n={len(df_gt)})', fontsize=14, pad=15)
    plt.xlabel('Count of Reviews', fontsize=12)
    plt.ylabel('Target Aspect', fontsize=12)
    plt.legend(title='Sentiment')
    
    plt.tight_layout()
    output_img = 'TargetAspect_vs_Sentiment_BarChart.png'
    plt.savefig(output_img, dpi=300)
    print(f"\nBar chart successfully saved as: {output_img}")
    plt.show()

if __name__ == "__main__":
    main()
