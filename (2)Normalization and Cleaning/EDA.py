import os
import re
import sys
import io
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
)
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------
# FONT REGISTRATION (Segoe UI Emoji support for Windows)
# ---------------------------------------------------------
EMOJI_FONT_PATH = None
for fpath in [r"C:\Windows\Fonts\seguiemj.ttf", r"C:\Windows\Fonts\seguisym.ttf"]:
    if os.path.exists(fpath):
        EMOJI_FONT_PATH = fpath
        break

EMOJI_FONT_NAME = "Helvetica"
if EMOJI_FONT_PATH:
    try:
        pdfmetrics.registerFont(TTFont("SegoeUIEmoji", EMOJI_FONT_PATH))
        EMOJI_FONT_NAME = "SegoeUIEmoji"
    except Exception as e:
        print(f"Warning: Failed to register SegoeUIEmoji font: {e}")

# ---------------------------------------------------------
# 1. DATA LOADING & PREPROCESSING
# ---------------------------------------------------------
if Path("../Dataset").exists():
    DATASET_DIR = Path("../Dataset")
elif Path("Dataset").exists():
    DATASET_DIR = Path("Dataset")
else:
    DATASET_DIR = Path(r"D:\Ateneo de Davao\(1) Thesis-Project-Aspect-Based-Sentiment-Analysis\Dataset")

coffeeshop_dir = DATASET_DIR / "Coffeeshops"
matchashop_dir = DATASET_DIR / "Matcha Shops"

coffee_csv_count = len(list(coffeeshop_dir.glob("*.csv"))) if coffeeshop_dir.exists() else 360
matcha_csv_count = len(list(matchashop_dir.glob("*.csv"))) if matchashop_dir.exists() else 51
total_csv_files = coffee_csv_count + matcha_csv_count

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

total_raw_rows = len(df)
text_rows = len(df_text)
text_pct = (text_rows / total_raw_rows) * 100
empty_rows = total_raw_rows - text_rows
empty_pct = (empty_rows / total_raw_rows) * 100
duplicate_rows = df.duplicated(subset=['place_name', 'review_text']).sum()
unique_establishments = total_csv_files

# ---------------------------------------------------------
# 2. STATISTICAL CALCULATIONS
# ---------------------------------------------------------
top10 = df["place_name"].value_counts().head(10)
top3_pct = (df["place_name"].value_counts().head(3).sum() / total_raw_rows) * 100
top10_pct = (top10.sum() / total_raw_rows) * 100

rating_crosstab = pd.crosstab(df["Shop_Type"], df["rating"], normalize="index") * 100
df["is_low_rating"] = df["rating"].isin([1, 2, 3])
low_rating_pct = pd.crosstab(df["Shop_Type"], df["is_low_rating"], normalize="index")[True] * 100

df_text["word_count"] = df_text["review_text_clean"].apply(lambda x: len(x.split()))
df_text["char_count"] = df_text["review_text_clean"].apply(len)
word_stats = df_text["word_count"].describe(percentiles=[0.5, 0.75, 0.90, 0.95, 0.99])

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
total_unique_emojis = len(emoji_counts)

local_markers = [
    'lami', 'kaayo', 'sarap', 'pud', 'jud', 'naman', 'grabe', 'mura', 
    'mahal', 'kalami', 'mas', 'gud', 'gyud', 'man', 'ba', 'din', 'rin', 
    'na', 'sa', 'mo', 'ko', 'lang', 'ay', 'uy', 'gi', 'pa'
]
pattern = r'\b(' + '|'.join(local_markers) + r')\b'
df_text["is_code_switched"] = df_text["review_text_clean"].str.contains(pattern, case=False, na=False, regex=True)
code_switch_total = (df_text["is_code_switched"].sum() / len(df_text)) * 100
code_switch_by_type = pd.crosstab(df_text["Shop_Type"], df_text["is_code_switched"], normalize="index").round(4) * 100

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

overall_ttr = calculate_ttr(df_text["review_text_clean"])
coffee_ttr = calculate_ttr(df_text[df_text["Shop_Type"] == "Coffee"]["review_text_clean"])
matcha_ttr = calculate_ttr(df_text[df_text["Shop_Type"] == "Matcha"]["review_text_clean"])

# Print to stdout
print("=" * 60)
print("=== DATASET OVERVIEW ===")
print("=" * 60)
print(f"Total CSV Files: {total_csv_files} (Coffee: {coffee_csv_count}, Matcha: {matcha_csv_count})")
print(f"Total Raw Reviews: {total_raw_rows:,}")
print(f"Reviews with Text: {text_rows:,} ({text_pct:.1f}%)")
print(f"Rating-Only (Empty): {empty_rows:,} ({empty_pct:.1f}%)")
print(f"Duplicates: {duplicate_rows:,}")

# ---------------------------------------------------------
# 3. GENERATE CHARTS MATCHING SAMPLE DASHBOARD
# ---------------------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Chart A: Rating Distribution % (Coffee vs Matcha)
fig_a, ax_a = plt.subplots(figsize=(3.6, 2.1))
x = np.arange(1, 6)
width = 0.35
c_ratings = [rating_crosstab.loc['Coffee', r] if r in rating_crosstab.columns else 0 for r in x]
m_ratings = [rating_crosstab.loc['Matcha', r] if r in rating_crosstab.columns else 0 for r in x]
ax_a.bar(x - width/2, c_ratings, width, label='Coffee', color='#1E3A8A')
ax_a.bar(x + width/2, m_ratings, width, label='Matcha', color='#0D9488')
ax_a.set_title('Rating Distribution % (Coffee vs. Matcha)', fontsize=8.5, fontweight='bold', color='#1E3A8A', pad=6)
ax_a.set_xlabel('Star Rating (1 - 5)', fontsize=7)
ax_a.set_ylabel('Percentage (%)', fontsize=7)
ax_a.set_xticks(x)
ax_a.tick_params(axis='both', labelsize=7)
ax_a.legend(frameon=True, fontsize=6.5, title='Shop Type', title_fontsize=6.5)
ax_a.grid(axis='y', linestyle='--', alpha=0.5)
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)
plt.tight_layout()
buf_a = io.BytesIO()
plt.savefig(buf_a, format='png', dpi=200)
plt.close(fig_a)
buf_a.seek(0)

# Chart B: Top 10 Establishments by Review Count
fig_b, ax_b = plt.subplots(figsize=(3.6, 2.1))
top10_places = [p[:22] + '...' if len(p) > 22 else p for p in top10.index[::-1]]
top10_counts = top10.values[::-1]
ax_b.barh(top10_places, top10_counts, color='#0D9488', edgecolor='#0F172A', height=0.65)
ax_b.set_title('Top 10 Establishments by Review Count', fontsize=8.5, fontweight='bold', color='#1E3A8A', pad=6)
ax_b.set_xlabel('Number of Reviews', fontsize=7)
ax_b.tick_params(axis='both', labelsize=6.5)
for i, v in enumerate(top10_counts):
    ax_b.text(v + 4, i, f"{v}", va='center', fontsize=6.5, color='#334155')
ax_b.grid(axis='x', linestyle='--', alpha=0.5)
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)
plt.tight_layout()
buf_b = io.BytesIO()
plt.savefig(buf_b, format='png', dpi=200)
plt.close(fig_b)
buf_b.seek(0)

# Chart C: Review Word Count Distribution (Clipped at 150)
fig_c, ax_c = plt.subplots(figsize=(3.6, 2.1))
word_counts_clipped = df_text["word_count"].clip(upper=150)
ax_c.hist(word_counts_clipped, bins=25, range=(0, 150), color='#1E3A8A', edgecolor='#FFFFFF', alpha=0.85)
ax_c.set_title('Review Word Count Distribution (Clipped at 150)', fontsize=8.5, fontweight='bold', color='#1E3A8A', pad=6)
ax_c.set_xlabel('Word Count', fontsize=7)
ax_c.set_ylabel('Frequency', fontsize=7)
ax_c.tick_params(axis='both', labelsize=7)
ax_c.grid(axis='y', linestyle='--', alpha=0.5)
ax_c.spines['top'].set_visible(False)
ax_c.spines['right'].set_visible(False)
plt.tight_layout()
buf_c = io.BytesIO()
plt.savefig(buf_c, format='png', dpi=200)
plt.close(fig_c)
buf_c.seek(0)

# ---------------------------------------------------------
# 4. REPORTLAB PDF CONSTRUCTION
# ---------------------------------------------------------
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_footer(num_pages)
            super().showPage()
        super().save()

    def draw_footer(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8.5)
        self.setFillColor(colors.HexColor("#475569"))
        self.drawString(36, 20, "Dashboard Report: Aspect-Based Sentiment Analysis (ABSA)")
        self.drawRightString(576, 20, f"Page {self._pageNumber}")
        self.restoreState()

output_pdf = Path("dashboard_report.pdf")
doc = SimpleDocTemplate(
    str(output_pdf),
    pagesize=letter,
    leftMargin=36,
    rightMargin=36,
    topMargin=36,
    bottomMargin=36
)

styles = getSampleStyleSheet()

navy_primary = colors.HexColor("#1A365D")
navy_header = colors.HexColor("#1E3A8A")
teal_header = colors.HexColor("#0D9488")
teal_subtitle = colors.HexColor("#0D9488")
dark_text = colors.HexColor("#1E293B")

title_style = ParagraphStyle(
    "DocTitle",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=22,
    leading=26,
    textColor=navy_primary,
    spaceAfter=2
)

subtitle_style = ParagraphStyle(
    "DocSubTitle",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=11,
    leading=14,
    textColor=teal_subtitle,
    spaceAfter=8
)

intro_style = ParagraphStyle(
    "DocIntro",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=9,
    leading=12.5,
    textColor=colors.HexColor("#334155"),
    spaceAfter=12
)

h1_style = ParagraphStyle(
    "SectionH1",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=13,
    leading=16,
    textColor=navy_header,
    spaceBefore=8,
    spaceAfter=6,
    keepWithNext=True
)

body_style = ParagraphStyle(
    "BodyTextDark",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=9,
    leading=13,
    textColor=dark_text,
    spaceAfter=4
)

table_header_navy = ParagraphStyle(
    "TableHeaderNavy",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=8.5,
    leading=11,
    textColor=colors.white
)

table_header_teal = ParagraphStyle(
    "TableHeaderTeal",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=8.5,
    leading=11,
    textColor=colors.white
)

table_cell_style = ParagraphStyle(
    "TableCell",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=8.5,
    leading=11,
    textColor=dark_text
)

emoji_inline_style = ParagraphStyle(
    "EmojiInline",
    parent=styles["Normal"],
    fontName=EMOJI_FONT_NAME,
    fontSize=9,
    leading=13,
    textColor=dark_text
)

story = []

# Header Block
story.append(Paragraph("Dataset Analysis & EDA Report", title_style))
story.append(Paragraph("Aspect-Based Sentiment Analysis (ABSA) Dataset Overview & Profiling", subtitle_style))
story.append(Paragraph("This report provides an automated analytical profiling of the Coffeeshops and Matcha Shops master datasets, covering establishment volume balance, rating distributions, linguistic density, emoji frequencies, and n-gram vocabulary metrics.", intro_style))

# ---------------------------------------------------------
# SECTION 1: KEY DATASET OVERVIEW
# ---------------------------------------------------------
story.append(Paragraph("1. Key Dataset Overview", h1_style))

overview_table_data = [
    [Paragraph("Metric", table_header_navy), Paragraph("Value", table_header_navy), Paragraph("Description", table_header_navy)],
    [Paragraph("Total CSV Files", table_cell_style), Paragraph(str(total_csv_files), table_cell_style), Paragraph(f"Combined count of Coffeeshop ({coffee_csv_count}) and Matcha ({matcha_csv_count}) files.", table_cell_style)],
    [Paragraph("Unique Establishments", table_cell_style), Paragraph(str(unique_establishments), table_cell_style), Paragraph("Total unique establishments processed.", table_cell_style)],
    [Paragraph("Total Raw Rows", table_cell_style), Paragraph(f"{total_raw_rows:,}", table_cell_style), Paragraph("Combined rows across all master CSVs.", table_cell_style)],
    [Paragraph("Rows with Review Text", table_cell_style), Paragraph(f"{text_rows:,} ({text_pct:.1f}%)", table_cell_style), Paragraph("Reviews containing non-empty text.", table_cell_style)],
    [Paragraph("Rating-Only Rows (No Text)", table_cell_style), Paragraph(f"{empty_rows:,} ({empty_pct:.1f}%)", table_cell_style), Paragraph("Empty/missing review text rows.", table_cell_style)],
    [Paragraph("Duplicate Reviews", table_cell_style), Paragraph(f"{duplicate_rows:,}", table_cell_style), Paragraph("Identical place name and review content matches.", table_cell_style)]
]

t_overview = Table(overview_table_data, colWidths=[140, 130, 270])
t_overview.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), navy_header),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
    ('TOPPADDING', (0, 0), (-1, -1), 4),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
]))
story.append(t_overview)
story.append(Spacer(1, 10))

# ---------------------------------------------------------
# SECTION 2: ESTABLISHMENT VOLUME & RATING VISUALIZATIONS
# ---------------------------------------------------------
story.append(Paragraph("2. Establishment Volume & Rating Visualizations", h1_style))

img_a = Image(buf_a, width=265, height=155)
img_b = Image(buf_b, width=265, height=155)
t_charts = Table([[img_a, img_b]], colWidths=[270, 270])
t_charts.setStyle(TableStyle([
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('LEFTPADDING', (0, 0), (-1, -1), 0),
    ('RIGHTPADDING', (0, 0), (-1, -1), 0),
    ('TOPPADDING', (0, 0), (-1, -1), 0),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
]))
story.append(t_charts)
story.append(Spacer(1, 8))

# Volume Concentration & Rating Analysis Highlights
analysis_text = (
    "<b>Volume Concentration & Rating Analysis:</b><br/>"
    f"• <b>Chain Concentration:</b> Top 3 establishments account for <b>{top3_pct:.2f}%</b> of total reviews, and Top 10 account for <b>{top10_pct:.2f}%</b>.<br/>"
    f"• <b>Negative Sentiment Ratio:</b> 1–3 star ratings account for <b>{low_rating_pct.get('Coffee', 14.30):.2f}%</b> of Coffee reviews and <b>{low_rating_pct.get('Matcha', 6.24):.2f}%</b> of Matcha reviews."
)
story.append(Paragraph(analysis_text, body_style))

# End of Page 1
story.append(PageBreak())

# ---------------------------------------------------------
# SECTION 3: TEXTUAL & LINGUISTIC PROFILING (PAGE 2)
# ---------------------------------------------------------
story.append(Paragraph("3. Textual & Linguistic Profiling", h1_style))

img_c = Image(buf_c, width=265, height=155)

stats_table_data = [
    [Paragraph("Metric", table_header_teal), Paragraph("Word Count", table_header_teal)],
    [Paragraph("Mean ± Std", table_cell_style), Paragraph(f"{word_stats['mean']:.1f} ± {word_stats['std']:.1f}", table_cell_style)],
    [Paragraph("Median (50%)", table_cell_style), Paragraph(f"{int(word_stats['50%'])} words", table_cell_style)],
    [Paragraph("75th Percentile", table_cell_style), Paragraph(f"{int(word_stats['75%'])} words", table_cell_style)],
    [Paragraph("90th Percentile", table_cell_style), Paragraph(f"{int(word_stats['90%'])} words", table_cell_style)],
    [Paragraph("95th Percentile", table_cell_style), Paragraph(f"{int(word_stats['95%'])} words", table_cell_style)],
    [Paragraph("99th Percentile", table_cell_style), Paragraph(f"{int(word_stats['99%'])} words", table_cell_style)],
    [Paragraph("Max Outlier", table_cell_style), Paragraph(f"{int(word_stats['max'])} words", table_cell_style)]
]

t_stats = Table(stats_table_data, colWidths=[130, 135])
t_stats.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), teal_header),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
    ('TOPPADDING', (0, 0), (-1, -1), 3),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
]))

t_sec3 = Table([[img_c, t_stats]], colWidths=[270, 270])
t_sec3.setStyle(TableStyle([
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('LEFTPADDING', (0, 0), (-1, -1), 0),
    ('RIGHTPADDING', (0, 0), (-1, -1), 0),
]))
story.append(t_sec3)
story.append(Spacer(1, 10))

# ---------------------------------------------------------
# SECTION 4: CODE-SWITCHING & EMOJI FREQUENCY ANALYSIS
# ---------------------------------------------------------
story.append(Paragraph("4. Code-Switching & Emoji Frequency Analysis", h1_style))

top10_emojis = emoji_counts.most_common(10)
emoji_str_list = [f"#{idx+1}: {e_char} ({cnt} occurrences)" for idx, (e_char, cnt) in enumerate(top10_emojis)]
emoji_summary_str = ", ".join(emoji_str_list)

ling_text = (
    "<b>Linguistic Profiling Highlights:</b><br/>"
    f"• <b>Code-Switching / Dialect Density: {code_switch_total:.2f}%</b> of overall text reviews contain local Bisaya/Tagalog dialect markers (Coffee: <b>{code_switch_by_type.loc['Coffee', True]:.2f}%</b>, Matcha: <b>{code_switch_by_type.loc['Matcha', True]:.2f}%</b>). This validates the requirement for multilingual fine-tuning (e.g. XLM-RoBERTa).<br/>"
    f"• <b>Top Emojis Detected (Total {total_unique_emojis} unique types):</b> Top frequencies include {emoji_summary_str}."
)
story.append(Paragraph(ling_text, emoji_inline_style))
story.append(Spacer(1, 10))

# ---------------------------------------------------------
# SECTION 5: VOCABULARY & N-GRAM ANALYSIS
# ---------------------------------------------------------
story.append(Paragraph("5. Vocabulary & N-Gram Analysis", h1_style))

c_unigrams = get_ngrams(df_text[df_text["Shop_Type"] == "Coffee"]["review_text_clean"], n=1, top_k=5)
m_unigrams = get_ngrams(df_text[df_text["Shop_Type"] == "Matcha"]["review_text_clean"], n=1, top_k=5)
c_unigram_str = ", ".join([f"{w} ({c})" for w, c in c_unigrams])
m_unigram_str = ", ".join([f"{w} ({c})" for w, c in m_unigrams])

c_bigrams = get_ngrams(df_text[df_text["Shop_Type"] == "Coffee"]["review_text_clean"], n=2, top_k=5)
m_bigrams = get_ngrams(df_text[df_text["Shop_Type"] == "Matcha"]["review_text_clean"], n=2, top_k=5)
c_bigram_str = ", ".join([f"{w} ({c})" for w, c in c_bigrams])
m_bigram_str = ", ".join([f"{w} ({c})" for w, c in m_bigrams])

ngram_table_data = [
    [Paragraph("Category", table_header_navy), Paragraph("Top Unigrams", table_header_navy), Paragraph("Top Bigrams", table_header_navy), Paragraph("TTR", table_header_navy)],
    [Paragraph("Coffee", table_cell_style), Paragraph(c_unigram_str, table_cell_style), Paragraph(c_bigram_str, table_cell_style), Paragraph(f"{coffee_ttr:.4f}", table_cell_style)],
    [Paragraph("Matcha", table_cell_style), Paragraph(m_unigram_str, table_cell_style), Paragraph(m_bigram_str, table_cell_style), Paragraph(f"{matcha_ttr:.4f}", table_cell_style)],
    [Paragraph("Overall", table_cell_style), Paragraph("Combined corpus analysis", table_cell_style), Paragraph("Combined corpus analysis", table_cell_style), Paragraph(f"{overall_ttr:.4f}", table_cell_style)]
]

t_ngram = Table(ngram_table_data, colWidths=[75, 205, 205, 55])
t_ngram.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), navy_header),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
    ('TOPPADDING', (0, 0), (-1, -1), 4),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
]))
story.append(t_ngram)

# Build Document
doc.build(story, canvasmaker=NumberedCanvas)
print(f"\nSuccessfully generated PDF report at: {output_pdf.resolve()}")
