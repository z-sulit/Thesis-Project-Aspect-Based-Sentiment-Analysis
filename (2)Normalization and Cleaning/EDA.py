import os
import re
import sys
import io
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import emoji
from sklearn.feature_extraction.text import CountVectorizer

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

# Font registration (Segoe UI Emoji support for Windows if available)
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
    except Exception:
        pass

# ---------------------------------------------------------
# DATA LOADING & PREPROCESSING
# ---------------------------------------------------------
base_path = Path("D:/Ateneo de Davao/Thesis-Project-Aspect-Based-Sentiment-Analysis/Dataset")
combined_file = base_path / "Thesis_master_reviews_combined.csv"

df = pd.read_csv(combined_file)
df.columns = df.columns.str.lower()

if 'clean_review' not in df.columns:
    df['clean_review'] = df['review_text'].fillna('')
if 'display_review' not in df.columns:
    df['display_review'] = df['review_text'].fillna('')

df['shop_type'] = df['shop_type'].astype(str).str.lower().str.capitalize()
df['place_name'] = df['place_name'].astype(str)

total_raw_rows = len(df)
coffee_rows = len(df[df['shop_type'] == 'Coffee'])
matcha_rows = len(df[df['shop_type'] == 'Matcha'])
unique_shops = df['place_name'].nunique()
empty_rows = df['clean_review'].str.strip().eq('').sum()
text_rows = total_raw_rows - empty_rows

# ---------------------------------------------------------
# STATISTICAL CALCULATIONS
# ---------------------------------------------------------
df['word_count'] = df['clean_review'].astype(str).str.split().str.len()
word_stats = df.groupby('shop_type')['word_count'].describe()

# N-Gram Analysis
def get_top_ngrams(corpus, n=2, top_k=5):
    vec = CountVectorizer(ngram_range=(n, n), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, int(sum_words[0, idx])) for word, idx in vec.vocabulary_.items()]
    return sorted(words_freq, key=lambda x: x[1], reverse=True)[:top_k]

coffee_bigrams = get_top_ngrams(df[df['shop_type'] == 'Coffee']['clean_review'].dropna(), n=2, top_k=5)
matcha_bigrams = get_top_ngrams(df[df['shop_type'] == 'Matcha']['clean_review'].dropna(), n=2, top_k=5)

# Type-Token Ratio (TTR)
def calculate_ttr(text_series):
    full_text = " ".join(text_series.dropna())
    tokens = full_text.split()
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)

coffee_ttr = calculate_ttr(df[df['shop_type'] == 'Coffee']['clean_review'])
matcha_ttr = calculate_ttr(df[df['shop_type'] == 'Matcha']['clean_review'])
overall_ttr = calculate_ttr(df['clean_review'])

# Code-Switching Density
strict_markers = ['jud', 'pud', 'sad', 'man', 'na', 'pa', 'din', 'rin']
strict_pattern = r'\b(' + '|'.join(strict_markers) + r')\b'
root_markers = ['lami', 'kaayo', 'sarap', 'mahal', 'init', 'tugnaw', 'balik', 'grabe', 'mura', 'pangit', 'bati']
root_pattern = r'(' + '|'.join(root_markers) + r')'
combined_pattern = f"({strict_pattern}|{root_pattern})"

df['code_switched'] = df['clean_review'].astype(str).str.contains(combined_pattern, case=False, na=False)
cs_by_type = df.groupby('shop_type')['code_switched'].mean() * 100

# Emoji Frequency
def extract_emojis(text):
    if pd.isna(text): return []
    return [res['emoji'] for res in emoji.emoji_list(str(text))]

all_emojis = [e for text in df['display_review'] for e in extract_emojis(text)]
emoji_counts = Counter(all_emojis)

# Console Output
print("=== Textual Profiling ===")
print(word_stats)
print("\n=== Top Bigrams (Coffee) ===")
print(coffee_bigrams)
print("\n=== Top Bigrams (Matcha) ===")
print(matcha_bigrams)
print(f"\n=== TTR ===\nCoffee: {coffee_ttr:.4f}, Matcha: {matcha_ttr:.4f}")
print(f"\n=== Code-Switching ===\n{cs_by_type}")

# ---------------------------------------------------------
# GENERATE CHARTS FOR PDF
# ---------------------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
sns.countplot(data=df, x='shop_type', palette=['#1E3A8A', '#0D9488'], ax=axes[0])
axes[0].set_title('Review Volume by Shop Type', fontsize=9, fontweight='bold', color='#1E3A8A')
axes[0].set_xlabel('Shop Type', fontsize=8)
axes[0].set_ylabel('Review Count', fontsize=8)

sns.histplot(data=df, x='rating', hue='shop_type', multiple='dodge', bins=5, palette=['#1E3A8A', '#0D9488'], ax=axes[1])
axes[1].set_title('Rating Distribution (1 - 5 Stars)', fontsize=9, fontweight='bold', color='#1E3A8A')
axes[1].set_xlabel('Star Rating', fontsize=8)
axes[1].set_ylabel('Frequency', fontsize=8)
plt.tight_layout()

chart_buf = io.BytesIO()
plt.savefig(chart_buf, format='png', dpi=200)
plt.close(fig)
chart_buf.seek(0)

# ---------------------------------------------------------
# REPORTLAB PDF GENERATION
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
        self.drawString(36, 20, "Aspect-Based Sentiment Analysis (ABSA) - Dataset EDA Report")
        self.drawRightString(576, 20, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()

output_pdf = base_path.parent / "eda_report.pdf"
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
dark_text = colors.HexColor("#1E293B")

title_style = ParagraphStyle("TitleStyle", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=20, leading=24, textColor=navy_primary, spaceAfter=4)
subtitle_style = ParagraphStyle("SubTitleStyle", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=10.5, leading=13, textColor=teal_header, spaceAfter=8)
h1_style = ParagraphStyle("H1Style", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=12, leading=15, textColor=navy_header, spaceBefore=8, spaceAfter=6)
cell_bold = ParagraphStyle("CellBold", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=8, leading=10, textColor=colors.white)
cell_style = ParagraphStyle("CellStyle", parent=styles["Normal"], fontName="Helvetica", fontSize=8, leading=10, textColor=dark_text)
emoji_cell_style = ParagraphStyle("EmojiCellStyle", parent=styles["Normal"], fontName=EMOJI_FONT_NAME, fontSize=8, leading=10, textColor=dark_text)

story = []

# Header
story.append(Paragraph("Dataset Exploratory Data Analysis (EDA)", title_style))
story.append(Paragraph("Coffeeshops vs. Matcha Shops Comparative Tabular Profiling", subtitle_style))
story.append(Spacer(1, 4))

# 1. Dataset Overview Table
story.append(Paragraph("1. Dataset Overview Summary", h1_style))
overview_data = [
    [Paragraph("Metric Description", cell_bold), Paragraph("Coffee Shops", cell_bold), Paragraph("Matcha Shops", cell_bold), Paragraph("Combined Corpus", cell_bold)],
    [Paragraph("Total Master Reviews", cell_style), Paragraph(f"{coffee_rows:,}", cell_style), Paragraph(f"{matcha_rows:,}", cell_style), Paragraph(f"{total_raw_rows:,}", cell_style)],
    [Paragraph("Reviews with Text", cell_style), Paragraph(f"{len(df[(df['shop_type']=='Coffee') & (df['clean_review']!='')]):,}", cell_style), Paragraph(f"{len(df[(df['shop_type']=='Matcha') & (df['clean_review']!='')]):,}", cell_style), Paragraph(f"{text_rows:,}", cell_style)],
    [Paragraph("Rating-Only Reviews (Empty Text)", cell_style), Paragraph(f"{len(df[(df['shop_type']=='Coffee') & (df['clean_review']=='')]):,}", cell_style), Paragraph(f"{len(df[(df['shop_type']=='Matcha') & (df['clean_review']=='')]):,}", cell_style), Paragraph(f"{empty_rows:,}", cell_style)],
    [Paragraph("Unique Establishments", cell_style), Paragraph(f"{df[df['shop_type']=='Coffee']['place_name'].nunique()}", cell_style), Paragraph(f"{df[df['shop_type']=='Matcha']['place_name'].nunique()}", cell_style), Paragraph(f"{unique_shops}", cell_style)]
]
t_overview = Table(overview_data, colWidths=[180, 120, 120, 120])
t_overview.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), navy_header),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
    ('PADDING', (0, 0), (-1, -1), 4),
]))
story.append(t_overview)
story.append(Spacer(1, 8))

# 2. Charts
story.append(Paragraph("2. Distribution Visualizations", h1_style))
img_chart = Image(chart_buf, width=540, height=185)
story.append(img_chart)
story.append(Spacer(1, 8))

# 3. Textual Profiling Table
story.append(Paragraph("3. Review Length & Word Count Profiling", h1_style))
word_table_data = [
    [Paragraph("Shop Type", cell_bold), Paragraph("Count", cell_bold), Paragraph("Mean ± Std", cell_bold), Paragraph("Median (50%)", cell_bold), Paragraph("75th Pct", cell_bold), Paragraph("Max Outlier", cell_bold)],
    [Paragraph("Coffee", cell_style), Paragraph(f"{int(word_stats.loc['Coffee','count']):,}", cell_style), Paragraph(f"{word_stats.loc['Coffee','mean']:.1f} ± {word_stats.loc['Coffee','std']:.1f}", cell_style), Paragraph(f"{int(word_stats.loc['Coffee','50%'])} words", cell_style), Paragraph(f"{int(word_stats.loc['Coffee','75%'])} words", cell_style), Paragraph(f"{int(word_stats.loc['Coffee','max'])} words", cell_style)],
    [Paragraph("Matcha", cell_style), Paragraph(f"{int(word_stats.loc['Matcha','count']):,}", cell_style), Paragraph(f"{word_stats.loc['Matcha','mean']:.1f} ± {word_stats.loc['Matcha','std']:.1f}", cell_style), Paragraph(f"{int(word_stats.loc['Matcha','50%'])} words", cell_style), Paragraph(f"{int(word_stats.loc['Matcha','75%'])} words", cell_style), Paragraph(f"{int(word_stats.loc['Matcha','max'])} words", cell_style)]
]
t_word = Table(word_table_data, colWidths=[90, 80, 110, 90, 85, 85])
t_word.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), teal_header),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
    ('PADDING', (0, 0), (-1, -1), 4),
]))
story.append(t_word)
story.append(Spacer(1, 8))

story.append(PageBreak())

# 4. Vocabulary & N-Gram Analysis Table
story.append(Paragraph("4. Vocabulary, Bigrams & Linguistic Metrics", h1_style))

c_bigram_str = ", ".join([f"{w} ({c})" for w, c in coffee_bigrams])
m_bigram_str = ", ".join([f"{w} ({c})" for w, c in matcha_bigrams])

ngram_table_data = [
    [Paragraph("Category", cell_bold), Paragraph("Top Bigrams (Count)", cell_bold), Paragraph("Type-Token Ratio (TTR)", cell_bold), Paragraph("Code-Switching %", cell_bold)],
    [Paragraph("Coffee", cell_style), Paragraph(c_bigram_str, cell_style), Paragraph(f"{coffee_ttr:.4f}", cell_style), Paragraph(f"{cs_by_type.get('Coffee', 0):.2f}%", cell_style)],
    [Paragraph("Matcha", cell_style), Paragraph(m_bigram_str, cell_style), Paragraph(f"{matcha_ttr:.4f}", cell_style), Paragraph(f"{cs_by_type.get('Matcha', 0):.2f}%", cell_style)],
    [Paragraph("Overall", cell_style), Paragraph("Combined corpus bigram profiling", cell_style), Paragraph(f"{overall_ttr:.4f}", cell_style), Paragraph(f"{df['code_switched'].mean()*100:.2f}%", cell_style)]
]

t_ngram = Table(ngram_table_data, colWidths=[70, 260, 110, 100])
t_ngram.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), navy_header),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
    ('PADDING', (0, 0), (-1, -1), 4),
]))
story.append(t_ngram)
story.append(Spacer(1, 10))

# 5. Top 10 Emojis Table
story.append(Paragraph("5. Top 10 Frequency Emoji Analysis Table", h1_style))
top10_emojis = emoji_counts.most_common(10)

emoji_table_data = [
    [Paragraph("Rank", cell_bold), Paragraph("Emoji", cell_bold), Paragraph("Demojized Name", cell_bold), Paragraph("Occurrences", cell_bold)]
]

for idx, (em_char, count) in enumerate(top10_emojis, 1):
    demoj_name = emoji.demojize(em_char)
    emoji_table_data.append([
        Paragraph(f"#{idx}", cell_style),
        Paragraph(em_char, emoji_cell_style),
        Paragraph(demoj_name, cell_style),
        Paragraph(f"{count:,}", cell_style)
    ])

t_emoji = Table(emoji_table_data, colWidths=[50, 60, 270, 160])
t_emoji.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), teal_header),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
    ('PADDING', (0, 0), (-1, -1), 3),
]))
story.append(t_emoji)

doc.build(story, canvasmaker=NumberedCanvas)
print(f"\nSUCCESS: Generated PDF Report at: {output_pdf.resolve()}")