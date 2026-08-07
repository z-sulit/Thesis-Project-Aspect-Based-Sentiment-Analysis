import os
import sys
import re
import glob
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Ensure UTF-8 output encoding for Windows terminal
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------
# 1. LOAD DATASET
# ---------------------------------------------------------
if Path("../Dataset").exists():
    DATASET_DIR = Path("../Dataset")
elif Path("Dataset").exists():
    DATASET_DIR = Path("Dataset")
else:
    DATASET_DIR = Path(r"D:\Ateneo de Davao\(1) Thesis-Project-Aspect-Based-Sentiment-Analysis\Dataset")

coffee_file = DATASET_DIR / "coffeeshops_master.csv"
matcha_file = DATASET_DIR / "matcha_shops_master.csv"

dfs = []
if coffee_file.exists():
    df_c = pd.read_csv(coffee_file)
    if "Shop_Type" not in df_c.columns or df_c["Shop_Type"].isnull().all():
        df_c["Shop_Type"] = "Coffee"
    dfs.append(df_c)

if matcha_file.exists():
    df_m = pd.read_csv(matcha_file)
    if "Shop_Type" not in df_m.columns or df_m["Shop_Type"].isnull().all():
        df_m["Shop_Type"] = "Matcha"
    dfs.append(df_m)

if not dfs:
    raise FileNotFoundError("Master CSV files not found under Dataset directory.")

df = pd.concat(dfs, ignore_index=True)
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df["review_text_clean"] = df["review_text"].fillna("").astype(str).str.strip()
df_text = df[df["review_text_clean"] != ""].copy()

# Calculate stats
coffeeshop_files = len(glob.glob(os.path.join(DATASET_DIR, "Coffeeshops", "*.csv")))
matcha_files = len(glob.glob(os.path.join(DATASET_DIR, "Matcha Shops", "*.csv")))
total_files = coffeeshop_files + matcha_files

top10 = df["place_name"].value_counts().head(10)
top3_pct = (df["place_name"].value_counts().head(3).sum() / len(df)) * 100
top10_pct = (top10.sum() / len(df)) * 100

rating_crosstab = pd.crosstab(df["Shop_Type"], df["rating"], normalize="index") * 100

df["is_low_rating"] = df["rating"].isin([1, 2, 3])
low_rating_pct = pd.crosstab(df["Shop_Type"], df["is_low_rating"], normalize="index")[True] * 100

df_text["word_count"] = df_text["review_text_clean"].apply(lambda x: len(x.split()))
df_text["char_count"] = df_text["review_text_clean"].apply(len)

length_stats = df_text[["word_count", "char_count"]].describe(percentiles=[0.25, 0.5, 0.75, 0.90, 0.95, 0.99])

# Emoji extraction
emoji_pattern = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002600-\U000026FF"
    "\U00002700-\U000027BF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA70-\U0001FAFF"
    "]+",
    flags=re.UNICODE,
)

all_emojis = []
for text in df_text["review_text_clean"]:
    found = emoji_pattern.findall(text)
    for e_group in found:
        all_emojis.extend(list(e_group))

emoji_counts = Counter(all_emojis)

# Code-switching
local_markers = [
    'lami', 'kaayo', 'sarap', 'pud', 'jud', 'naman', 'grabe', 'mura', 
    'mahal', 'kalami', 'mas', 'gud', 'gyud', 'man', 'ba', 'din', 'rin', 
    'na', 'sa', 'mo', 'ko', 'lang', 'ay', 'uy', 'gi', 'pa'
]
pattern = r'\b(?:' + '|'.join(local_markers) + r')\b'
df_text["is_code_switched"] = df_text["review_text_clean"].str.contains(pattern, case=False, na=False, regex=True)
code_switch_crosstab = pd.crosstab(df_text["Shop_Type"], df_text["is_code_switched"], normalize="index") * 100

# N-Grams
STOPWORDS = set([
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "up", "about", "into", "over", "after", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "it", "its", "this", "that", "they",
    "them", "their", "we", "us", "our", "you", "your", "i", "my", "me", "very", "so"
])

def get_ngrams(texts, n=1, top_k=5):
    words_list = []
    for text in texts:
        tokens = [w.lower() for w in re.findall(r'\b[a-zA-Z]{2,}\b', text) if w.lower() not in STOPWORDS]
        if n == 1:
            words_list.extend(tokens)
        elif n == 2:
            bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens)-1)]
            words_list.extend(bigrams)
    return Counter(words_list).most_common(top_k)

def calculate_ttr(texts):
    all_tokens = [w.lower() for text in texts for w in re.findall(r'\b[a-zA-Z]{2,}\b', text)]
    if not all_tokens:
        return 0.0
    return len(set(all_tokens)) / len(all_tokens)

print("=" * 60)
print("=== DATASET OVERVIEW & EDA ===")
print("=" * 60)
print(f"Total Raw Reviews: {len(df)}")
print(f"Reviews with Text: {len(df_text)} ({len(df_text)/len(df)*100:.1f}%)")
print(f"Rating-Only (Empty): {len(df) - len(df_text)} ({(len(df)-len(df_text))/len(df)*100:.1f}%)")

# ---------------------------------------------------------
# 2. GENERATE VISUALIZATIONS
# ---------------------------------------------------------
temp_dir = "./temp_charts"
os.makedirs(temp_dir, exist_ok=True)

colors_hex = {
    "primary": "#1E3A8A",
    "secondary": "#0D9488",
    "accent": "#F59E0B",
    "background": "#F3F4F6",
    "dark": "#1F2937",
    "light_teal": "#CCFBF1"
}

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["text.color"] = colors_hex["dark"]
plt.rcParams["axes.labelcolor"] = colors_hex["dark"]

# Chart 1: Rating Distribution Coffee vs Matcha
fig, ax = plt.subplots(figsize=(6, 3.2))
rating_df = pd.crosstab(df["rating"], df["Shop_Type"], normalize="columns") * 100
rating_df.plot(kind="bar", ax=ax, color=[colors_hex["primary"], colors_hex["secondary"]], width=0.7, edgecolor="black", linewidth=0.5)
ax.set_title("Rating Distribution % (Coffee vs. Matcha)", fontsize=11, fontweight="bold", color=colors_hex["primary"], pad=10)
ax.set_xlabel("Star Rating (1 - 5)", fontsize=9)
ax.set_ylabel("Percentage (%)", fontsize=9)
ax.legend(title="Shop Type", fontsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
chart_rating = os.path.join(temp_dir, "rating_dist.png")
plt.savefig(chart_rating, dpi=300)
plt.close()

# Chart 2: Top 10 Establishments
fig, ax = plt.subplots(figsize=(6, 3.2))
y_pos = np.arange(len(top10))
truncated_names = [name[:22] + "..." if len(name) > 22 else name for name in top10.index]
bars = ax.barh(y_pos, top10.values, color=colors_hex["secondary"], height=0.6, edgecolor=colors_hex["primary"], linewidth=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(truncated_names, fontsize=8)
ax.invert_yaxis()
ax.set_title("Top 10 Establishments by Review Count", fontsize=11, fontweight="bold", color=colors_hex["primary"], pad=10)
ax.set_xlabel("Number of Reviews", fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", linestyle="--", alpha=0.5)
for bar in bars:
    w = bar.get_width()
    ax.annotate(f"{int(w)}", xy=(w, bar.get_y() + bar.get_height()/2), xytext=(3, 0), textcoords="offset points", ha="left", va="center", fontsize=8)
plt.tight_layout()
chart_top_places = os.path.join(temp_dir, "top_places.png")
plt.savefig(chart_top_places, dpi=300)
plt.close()

# Chart 3: Word Count Distribution
fig, ax = plt.subplots(figsize=(6, 3))
words_clipped = df_text["word_count"].clip(upper=150)
ax.hist(words_clipped, bins=25, color=colors_hex["primary"], edgecolor="white", alpha=0.85)
ax.set_title("Review Word Count Distribution (Clipped at 150)", fontsize=11, fontweight="bold", color=colors_hex["primary"], pad=10)
ax.set_xlabel("Word Count", fontsize=9)
ax.set_ylabel("Frequency", fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
chart_word_dist = os.path.join(temp_dir, "word_dist.png")
plt.savefig(chart_word_dist, dpi=300)
plt.close()

# ---------------------------------------------------------
# 3. BUILD REPORTLAB PDF
# ---------------------------------------------------------
root_pdf_path = Path("dashboard_report.pdf")
if DATASET_DIR.parent.exists():
    root_pdf_path = DATASET_DIR.parent / "dashboard_report.pdf"

doc = SimpleDocTemplate(
    str(root_pdf_path),
    pagesize=letter,
    leftMargin=36, rightMargin=36,
    topMargin=36, bottomMargin=36
)

c_primary = colors.HexColor("#1E3A8A")
c_secondary = colors.HexColor("#0D9488")
c_dark = colors.HexColor("#1F2937")
c_light = colors.HexColor("#F3F4F6")
c_white = colors.white

styles = getSampleStyleSheet()

title_style = ParagraphStyle("DocTitle", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=22, leading=26, textColor=c_primary, spaceAfter=4)
subtitle_style = ParagraphStyle("DocSubTitle", parent=styles["Normal"], fontName="Helvetica", fontSize=10.5, leading=14, textColor=c_secondary, spaceAfter=12)
h1_style = ParagraphStyle("H1", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=13, leading=16, textColor=c_primary, spaceBefore=8, spaceAfter=6, keepWithNext=True)
body_style = ParagraphStyle("Body", parent=styles["Normal"], fontName="Helvetica", fontSize=9, leading=13, textColor=c_dark, spaceAfter=6)
table_text = ParagraphStyle("TT", parent=styles["Normal"], fontName="Helvetica", fontSize=8, leading=10, textColor=c_dark)
table_header = ParagraphStyle("TH", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=8, leading=10, textColor=c_white)

story = []

# Page 1 Header
story.append(Paragraph("Dataset Analysis & EDA Report", title_style))
story.append(Paragraph("Aspect-Based Sentiment Analysis (ABSA) Dataset Overview & Profiling", subtitle_style))
story.append(Paragraph("This report provides an automated analytical profiling of the Coffeeshops and Matcha Shops master datasets, covering establishment volume balance, rating distributions, linguistic density, emoji frequencies, and n-gram vocabulary metrics.", body_style))
story.append(Spacer(1, 8))

# Table 1: Key Metrics
story.append(Paragraph("1. Key Dataset Overview", h1_style))
stats_data = [
    [Paragraph("Metric", table_header), Paragraph("Value", table_header), Paragraph("Description", table_header)],
    [Paragraph("Total CSV Files", table_text), Paragraph(f"{total_files}", table_text), Paragraph("Combined count of Coffeeshop (360) and Matcha (51) files.", table_text)],
    [Paragraph("Unique Establishments", table_text), Paragraph(f"{total_files}", table_text), Paragraph("Total unique establishments processed.", table_text)],
    [Paragraph("Total Raw Rows", table_text), Paragraph(f"{len(df):,}", table_text), Paragraph("Combined rows across all master CSVs.", table_text)],
    [Paragraph("Rows with Review Text", table_text), Paragraph(f"{len(df_text):,} ({len(df_text)/len(df)*100:.1f}%)", table_text), Paragraph("Reviews containing non-empty text.", table_text)],
    [Paragraph("Rating-Only Rows (No Text)", table_text), Paragraph(f"{len(df)-len(df_text):,} ({(len(df)-len(df_text))/len(df)*100:.1f}%)", table_text), Paragraph("Empty/missing review text rows.", table_text)],
    [Paragraph("Duplicate Reviews", table_text), Paragraph(f"{df.duplicated(subset=['place_name', 'review_text']).sum():,}", table_text), Paragraph("Identical place name and review content matches.", table_text)],
]
stats_table = Table(stats_data, colWidths=[140, 100, 300])
stats_table.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), c_primary),
    ("ALIGN", (0,0), (-1,-1), "LEFT"),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("TOPPADDING", (0,0), (-1,-1), 3),
    ("BOTTOMPADDING", (0,0), (-1,-1), 3),
    ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#D1D5DB")),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [c_white, c_light])
]))
story.append(stats_table)
story.append(Spacer(1, 10))

# Visualizations side-by-side
story.append(Paragraph("2. Establishment Volume & Rating Visualizations", h1_style))
chart_table = Table([[Image(chart_rating, width=260, height=135), Image(chart_top_places, width=260, height=135)]], colWidths=[270, 270])
chart_table.setStyle(TableStyle([
    ("ALIGN", (0,0), (-1,-1), "CENTER"),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("LEFTPADDING", (0,0), (-1,-1), 0),
    ("RIGHTPADDING", (0,0), (-1,-1), 0),
]))
story.append(chart_table)
story.append(Spacer(1, 10))

# Section: Establishment Volume Concentration & Rating Crosstab
volume_summary = (
    f"<b>Volume Concentration & Rating Analysis:</b><br/>"
    f"• <b>Chain Concentration:</b> Top 3 establishments account for <b>{top3_pct:.2f}%</b> of total reviews, and Top 10 account for <b>{top10_pct:.2f}%</b>.<br/>"
    f"• <b>Negative Sentiment Ratio:</b> 1–3 star ratings account for <b>{low_rating_pct.get('Coffee', 0):.2f}%</b> of Coffee reviews and <b>{low_rating_pct.get('Matcha', 0):.2f}%</b> of Matcha reviews."
)
story.append(Paragraph(volume_summary, body_style))

# Page 2: Textual & Linguistic Profiling
story.append(PageBreak())
story.append(Paragraph("3. Textual & Linguistic Profiling", h1_style))

# Length Stats Table
length_data = [
    [Paragraph("Metric", table_header), Paragraph("Word Count", table_header), Paragraph("Character Count", table_header)],
    [Paragraph("Mean ± Std", table_text), Paragraph(f"{length_stats.loc['mean', 'word_count']:.1f} ± {length_stats.loc['std', 'word_count']:.1f}", table_text), Paragraph(f"{length_stats.loc['mean', 'char_count']:.1f} ± {length_stats.loc['std', 'char_count']:.1f}", table_text)],
    [Paragraph("Median (50%)", table_text), Paragraph(f"{length_stats.loc['50%', 'word_count']:.0f} words", table_text), Paragraph(f"{length_stats.loc['50%', 'char_count']:.0f} chars", table_text)],
    [Paragraph("75th Percentile", table_text), Paragraph(f"{length_stats.loc['75%', 'word_count']:.0f} words", table_text), Paragraph(f"{length_stats.loc['75%', 'char_count']:.0f} chars", table_text)],
    [Paragraph("90th Percentile", table_text), Paragraph(f"{length_stats.loc['90%', 'word_count']:.0f} words", table_text), Paragraph(f"{length_stats.loc['90%', 'char_count']:.0f} chars", table_text)],
    [Paragraph("95th Percentile", table_text), Paragraph(f"{length_stats.loc['95%', 'word_count']:.0f} words", table_text), Paragraph(f"{length_stats.loc['95%', 'char_count']:.0f} chars", table_text)],
    [Paragraph("99th Percentile", table_text), Paragraph(f"{length_stats.loc['99%', 'word_count']:.0f} words", table_text), Paragraph(f"{length_stats.loc['99%', 'char_count']:.0f} chars", table_text)],
    [Paragraph("Max Outlier", table_text), Paragraph(f"{length_stats.loc['max', 'word_count']:.0f} words", table_text), Paragraph(f"{length_stats.loc['max', 'char_count']:.0f} chars", table_text)],
]
length_table = Table(length_data, colWidths=[140, 180, 220])
length_table.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), c_secondary),
    ("ALIGN", (0,0), (-1,-1), "LEFT"),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("TOPPADDING", (0,0), (-1,-1), 3),
    ("BOTTOMPADDING", (0,0), (-1,-1), 3),
    ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#D1D5DB")),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [c_white, c_light])
]))

length_layout = Table([[Image(chart_word_dist, width=260, height=130), length_table]], colWidths=[270, 270])
length_layout.setStyle(TableStyle([
    ("ALIGN", (0,0), (-1,-1), "LEFT"),
    ("VALIGN", (0,0), (-1,-1), "TOP"),
    ("LEFTPADDING", (0,0), (-1,-1), 0),
    ("RIGHTPADDING", (0,0), (-1,-1), 0),
]))
story.append(length_layout)
story.append(Spacer(1, 10))

# Code-Switching & Emoji Table
story.append(Paragraph("4. Code-Switching & Emoji Frequency Analysis", h1_style))

# Top 10 Emojis formatted safely
top10_emojis = emoji_counts.most_common(10)
emoji_str_list = [f"#{idx+1}: {cnt} occurrences" for idx, (e, cnt) in enumerate(top10_emojis)]
emoji_summary_text = ", ".join(emoji_str_list) if emoji_str_list else "None detected"

code_switch_overall = (df_text["is_code_switched"].sum() / len(df_text)) * 100
coffee_cs = code_switch_crosstab.loc["Coffee", True] if "Coffee" in code_switch_crosstab.index else 0
matcha_cs = code_switch_crosstab.loc["Matcha", True] if "Matcha" in code_switch_crosstab.index else 0

linguistic_summary = (
    f"<b>Linguistic Profiling Highlights:</b><br/>"
    f"• <b>Code-Switching / Dialect Density:</b> <b>{code_switch_overall:.2f}%</b> of overall text reviews contain local Bisaya/Tagalog dialect markers "
    f"(Coffee: <b>{coffee_cs:.2f}%</b>, Matcha: <b>{matcha_cs:.2f}%</b>). This validates the requirement for multilingual fine-tuning (e.g. XLM-RoBERTa).<br/>"
    f"• <b>Top Emojis Detected (Total {len(emoji_counts):,} unique types):</b> Top frequencies include {emoji_summary_text}."
)
story.append(Paragraph(linguistic_summary, body_style))
story.append(Spacer(1, 10))

# Section 5: N-Gram & Vocabulary Richness
story.append(Paragraph("5. Vocabulary & N-Gram Analysis", h1_style))

coffee_unigrams = get_ngrams(df_text[df_text["Shop_Type"]=="Coffee"]["review_text_clean"], n=1, top_k=5)
matcha_unigrams = get_ngrams(df_text[df_text["Shop_Type"]=="Matcha"]["review_text_clean"], n=1, top_k=5)
coffee_bigrams = get_ngrams(df_text[df_text["Shop_Type"]=="Coffee"]["review_text_clean"], n=2, top_k=5)
matcha_bigrams = get_ngrams(df_text[df_text["Shop_Type"]=="Matcha"]["review_text_clean"], n=2, top_k=5)

c_ttr = calculate_ttr(df_text[df_text["Shop_Type"]=="Coffee"]["review_text_clean"])
m_ttr = calculate_ttr(df_text[df_text["Shop_Type"]=="Matcha"]["review_text_clean"])
o_ttr = calculate_ttr(df_text["review_text_clean"])

ngram_data = [
    [Paragraph("Category", table_header), Paragraph("Top Unigrams", table_header), Paragraph("Top Bigrams", table_header), Paragraph("TTR", table_header)],
    [
        Paragraph("<b>Coffee</b>", table_text),
        Paragraph(", ".join([f"{w} ({c})" for w,c in coffee_unigrams]), table_text),
        Paragraph(", ".join([f"{b} ({c})" for b,c in coffee_bigrams]), table_text),
        Paragraph(f"{c_ttr:.4f}", table_text)
    ],
    [
        Paragraph("<b>Matcha</b>", table_text),
        Paragraph(", ".join([f"{w} ({c})" for w,c in matcha_unigrams]), table_text),
        Paragraph(", ".join([f"{b} ({c})" for b,c in matcha_bigrams]), table_text),
        Paragraph(f"{m_ttr:.4f}", table_text)
    ],
    [
        Paragraph("<b>Overall</b>", table_text),
        Paragraph("Combined corpus analysis", table_text),
        Paragraph("Combined corpus analysis", table_text),
        Paragraph(f"<b>{o_ttr:.4f}</b>", table_text)
    ]
]
ngram_table = Table(ngram_data, colWidths=[70, 180, 220, 70])
ngram_table.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), c_primary),
    ("ALIGN", (0,0), (-1,-1), "LEFT"),
    ("VALIGN", (0,0), (-1,-1), "TOP"),
    ("TOPPADDING", (0,0), (-1,-1), 4),
    ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#D1D5DB")),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [c_white, c_light])
]))
story.append(ngram_table)

# Footer and Page Numbering
def add_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(c_dark)
    canvas.drawString(36, 20, "Dashboard Report: Aspect-Based Sentiment Analysis (ABSA)")
    canvas.drawRightString(letter[0] - 36, 20, f"Page {doc.page}")
    canvas.restoreState()

doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)

# Clean up temp charts
for p in [chart_rating, chart_top_places, chart_word_dist]:
    try:
        os.remove(p)
    except Exception:
        pass
try:
    os.rmdir(temp_dir)
except Exception:
    pass

print(f"Report successfully saved to: {root_pdf_path.resolve()}")
