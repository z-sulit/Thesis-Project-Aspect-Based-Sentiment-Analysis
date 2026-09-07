import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

np.random.seed(42)


def load_and_validate_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input file not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    required_columns = [
        "clean_review",
        "sentence",
        "TargetAspect",
        "shop_type",
        "rating",
    ]
    for col in required_columns:
        assert col in df.columns, f"Dataset missing required column: '{col}'"

    null_counts = df[required_columns].isnull().sum()
    assert null_counts.sum() == 0, f"Null values detected in required columns:\n{null_counts}"

    duplicated_records = df.duplicated().sum()
    print(f"Data validation passed: {len(df)} rows loaded successfully.")
    print(f"Duplicate rows detected: {duplicated_records}")

    return df


def analyze_aspect_class_imbalance(df: pd.DataFrame, output_dir: Path) -> None:
    counts = df["TargetAspect"].value_counts().sort_values(ascending=True)
    total = len(df)
    percentages = (counts / total) * 100

    print("\n" + "=" * 50)
    print("1. ASPECT CLASS IMBALANCE SUMMARY")
    print("=" * 50)
    summary_df = pd.DataFrame(
        {"Count": counts, "Percentage (%)": percentages.round(2)}
    ).sort_values(by="Count", ascending=False)
    print(summary_df.to_string())

    plt.figure(figsize=(10, 6))
    bars = plt.barh(counts.index, counts.values, color="#2b5c8f", edgecolor="black")
    plt.title("Target Aspect Class Frequency (Imbalance Analysis)", fontsize=14, pad=12)
    plt.xlabel("Frequency (Review-Aspect Pairs)", fontsize=11)
    plt.ylabel("Target Aspect", fontsize=11)
    plt.grid(axis="x", linestyle="--", alpha=0.6)

    max_x = counts.max() * 1.15
    plt.xlim(0, max_x)

    for bar in bars:
        width = bar.get_width()
        pct = (width / total) * 100
        plt.text(
            width + (max_x * 0.01),
            bar.get_y() + bar.get_height() / 2,
            f"{int(width):,} ({pct:.1f}%)",
            va="center",
            ha="left",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(output_dir / "1_aspect_class_imbalance.png", dpi=300)
    plt.close()


def analyze_sentence_length_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    word_counts = df["sentence"].astype(str).str.strip().apply(lambda s: len(s.split()))

    p50 = float(np.percentile(word_counts, 50))
    p90 = float(np.percentile(word_counts, 90))
    p95 = float(np.percentile(word_counts, 95))
    p99 = float(np.percentile(word_counts, 99))
    max_len = int(word_counts.max())

    print("\n" + "=" * 50)
    print("2. SENTENCE LENGTH DISTRIBUTION & PERCENTILES")
    print("=" * 50)
    print(f"Median (50th percentile) : {p50:.1f} words")
    print(f"90th Percentile           : {p90:.1f} words")
    print(f"95th Percentile           : {p95:.1f} words")
    print(f"99th Percentile           : {p99:.1f} words")
    print(f"Maximum sentence length   : {max_len} words")

    plt.figure(figsize=(10, 6))
    sns.histplot(
        word_counts,
        bins=min(50, max_len),
        kde=True,
        color="#3b7a57",
        edgecolor="black",
    )
    plt.axvline(
        p90,
        color="#d95f02",
        linestyle="--",
        linewidth=2,
        label=f"90th Percentile ({p90:.1f} words)",
    )
    plt.axvline(
        p95,
        color="#7570b3",
        linestyle="--",
        linewidth=2,
        label=f"95th Percentile ({p95:.1f} words)",
    )

    plt.title("Decomposed Sentence Word Count Distribution", fontsize=14, pad=12)
    plt.xlabel("Word Count per Sentence", fontsize=11)
    plt.ylabel("Frequency", fontsize=11)
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_dir / "2_sentence_length_distribution.png", dpi=300)
    plt.close()


def analyze_aspect_density_per_review(df: pd.DataFrame, output_dir: Path) -> None:
    aspect_density = df.groupby("clean_review")["TargetAspect"].nunique()
    counts = aspect_density.value_counts().sort_index()
    total_reviews = len(aspect_density)
    percentages = (counts / total_reviews) * 100

    print("\n" + "=" * 50)
    print("3. ASPECT DENSITY PER REVIEW")
    print("=" * 50)
    density_df = pd.DataFrame(
        {"Review Count": counts, "Percentage (%)": percentages.round(2)}
    )
    print(density_df.to_string())

    plt.figure(figsize=(9, 5.5))
    bars = plt.bar(
        counts.index.astype(str),
        counts.values,
        color="#e67e22",
        edgecolor="black",
        width=0.6,
    )
    plt.title("Aspect Density per Review (Unique Aspects Mentioned)", fontsize=14, pad=12)
    plt.xlabel("Number of Unique Aspects per Review", fontsize=11)
    plt.ylabel("Number of Reviews", fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    max_y = counts.max() * 1.15
    plt.ylim(0, max_y)

    for bar in bars:
        height = bar.get_height()
        pct = (height / total_reviews) * 100
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + (max_y * 0.015),
            f"{int(height):,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(output_dir / "3_aspect_density_per_review.png", dpi=300)
    plt.close()


def analyze_aspects_vs_shop_type(df: pd.DataFrame, output_dir: Path) -> None:
    df_clean = df.copy()
    df_clean["shop_type"] = df_clean["shop_type"].astype(str).str.strip().str.title()

    crosstab_raw = pd.crosstab(df_clean["TargetAspect"], df_clean["shop_type"])
    crosstab_norm = (
        pd.crosstab(
            df_clean["TargetAspect"], df_clean["shop_type"], normalize="columns"
        )
        * 100
    )

    print("\n" + "=" * 50)
    print("4. ASPECTS VS. SHOP TYPE CROSS-TABULATION")
    print("=" * 50)
    print("\nAbsolute Frequencies:")
    print(crosstab_raw.to_string())
    print("\nWithin-Shop Column Percentages (%):")
    print(crosstab_norm.round(2).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    crosstab_raw.plot(kind="barh", ax=axes[0], colormap="viridis", edgecolor="black")
    axes[0].set_title("Aspect Frequencies by Shop Type (Raw Counts)", fontsize=13)
    axes[0].set_xlabel("Count", fontsize=11)
    axes[0].set_ylabel("Target Aspect", fontsize=11)
    axes[0].grid(axis="x", linestyle="--", alpha=0.6)

    annot_data = crosstab_raw.astype(str) + "\n(" + crosstab_norm.round(1).astype(str) + "%)"
    sns.heatmap(
        crosstab_norm,
        annot=annot_data,
        fmt="",
        cmap="Blues",
        cbar_kws={"label": "Percentage of Shop Type Aspects (%)"},
        ax=axes[1],
        linewidths=0.5,
        linecolor="gray",
    )
    axes[1].set_title("Aspect Share by Shop Type (% of Shop Total)", fontsize=13)
    axes[1].set_xlabel("Shop Type", fontsize=11)
    axes[1].set_ylabel("")

    plt.tight_layout()
    plt.savefig(output_dir / "4_aspects_vs_shop_type.png", dpi=300)
    plt.close()


def analyze_rating_vs_target_aspect(df: pd.DataFrame, output_dir: Path) -> None:
    df_clean = df.copy()
    df_clean["rating"] = pd.to_numeric(df_clean["rating"], errors="coerce")
    assert df_clean["rating"].notnull().all(), "Non-numeric values found in rating column."

    rating_summary = (
        df_clean.groupby("TargetAspect")["rating"]
        .agg(["count", "mean", "median", "std"])
        .sort_values(by="mean", ascending=False)
    )

    print("\n" + "=" * 50)
    print("5. OVERALL RATING VS. TARGET ASPECT (SENTIMENT PROXY)")
    print("=" * 50)
    print(rating_summary.round(3).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    bars = axes[0].barh(
        rating_summary.index[::-1],
        rating_summary["mean"][::-1],
        color="#2980b9",
        edgecolor="black",
        xerr=rating_summary["std"][::-1],
        capsize=4,
    )
    axes[0].set_title("Mean Rating by Target Aspect (Error Bar: 1 SD)", fontsize=13)
    axes[0].set_xlabel("Mean Rating (1-5 Scale)", fontsize=11)
    axes[0].set_ylabel("Target Aspect", fontsize=11)
    axes[0].set_xlim(1.0, 5.2)
    axes[0].grid(axis="x", linestyle="--", alpha=0.6)

    for bar in bars:
        width = bar.get_width()
        axes[0].text(
            width + 0.08,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2f}",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
        )

    sns.boxplot(
        data=df_clean,
        x="rating",
        y="TargetAspect",
        hue="TargetAspect",
        legend=False,
        order=rating_summary.index,
        palette="Set2",
        ax=axes[1],
        orient="h",
    )
    axes[1].set_title("Rating Distribution per Target Aspect", fontsize=13)
    axes[1].set_xlabel("Rating", fontsize=11)
    axes[1].set_ylabel("")

    plt.tight_layout()
    plt.savefig(output_dir / "5_rating_vs_target_aspect.png", dpi=300)
    plt.close()


def main() -> None:
    current_dir = Path(__file__).resolve().parent
    csv_path = current_dir / "Exploded_Review_Aspect_Pairs.csv"
    output_dir = current_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_validate_data(csv_path)

    analyze_aspect_class_imbalance(df, output_dir)
    analyze_sentence_length_distribution(df, output_dir)
    analyze_aspect_density_per_review(df, output_dir)
    analyze_aspects_vs_shop_type(df, output_dir)
    analyze_rating_vs_target_aspect(df, output_dir)

    print(f"\nAll 5 EDA tasks completed successfully. Figures saved to: {output_dir}")


main()
