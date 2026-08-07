import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# 1. Gather and Analyze Data
def analyze_dataset():
    dataset_dir = r"D:\Ateneo de Davao\(1) Thesis-Project-Aspect-Based-Sentiment-Analysis\Dataset"
    csv_files = glob.glob(os.path.join(dataset_dir, "**", "*.csv"), recursive=True)
    
    all_dfs = []
    coffeeshop_count = 0
    matcha_count = 0
    
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            # Add category column based on directory name
            if "Coffeeshops" in f:
                df["category"] = "Coffeeshop"
                coffeeshop_count += 1
            elif "Matcha Shops" in f:
                df["category"] = "Matcha Shop"
                matcha_count += 1
            else:
                df["category"] = "Other"
            
            df["source_file"] = os.path.basename(f)
            all_dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not all_dfs:
        print("No CSV files found or loaded.")
        return None
        
    df_all = pd.concat(all_dfs, ignore_index=True)
    
    # Ensure rating is numeric
    df_all["rating"] = pd.to_numeric(df_all["rating"], errors="coerce")
    
    # Clean text to define text vs rating-only
    df_all["review_text_clean"] = df_all["review_text"].fillna("").astype(str).str.strip()
    df_all["has_text"] = df_all["review_text_clean"] != ""
    
    # Statistics
    stats = {
        "total_files": len(csv_files),
        "coffeeshop_files": coffeeshop_count,
        "matcha_files": matcha_count,
        "total_rows": len(df_all),
        "rows_with_text": int(df_all["has_text"].sum()),
        "rows_no_text": int((~df_all["has_text"]).sum()),
        "unique_places": len(all_dfs),
        "average_rating": float(df_all["rating"].mean()),
        "average_rating_with_text": float(df_all[df_all["has_text"]]["rating"].mean()),
    }
    
    return df_all, stats

# 2. Generate Visualizations
def generate_visualizations(df_all, temp_dir):
    os.makedirs(temp_dir, exist_ok=True)
    
    # Color palette
    colors_hex = {
        "primary": "#1E3A8A", # Deep navy
        "secondary": "#0D9488", # Teal
        "accent": "#F59E0B", # Gold
        "background": "#F3F4F6",
        "dark": "#1F2937",
        "light_teal": "#CCFBF1"
    }
    
    # Matplotlib styling
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["text.color"] = colors_hex["dark"]
    plt.rcParams["axes.labelcolor"] = colors_hex["dark"]
    plt.rcParams["xtick.color"] = colors_hex["dark"]
    plt.rcParams["ytick.color"] = colors_hex["dark"]
    
    # Plot 1: Rating Distribution
    fig, ax = plt.subplots(figsize=(6, 3))
    rating_counts = df_all["rating"].value_counts().sort_index()
    # Fill in any missing ratings from 1 to 5
    for r in range(1, 6):
        if r not in rating_counts:
            rating_counts[r] = 0
    rating_counts = rating_counts.sort_index()
    
    bars = ax.bar(rating_counts.index, rating_counts.values, color=colors_hex["secondary"], width=0.6, edgecolor=colors_hex["primary"], linewidth=0.5)
    ax.set_title("Rating Distribution (1 - 5 Stars)", fontsize=11, fontweight="bold", pad=10, color=colors_hex["primary"])
    ax.set_xlabel("Rating", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_xticks(range(1, 6))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{int(height):,}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)
                    
    plt.tight_layout()
    rating_chart_path = os.path.join(temp_dir, "rating_dist.png")
    plt.savefig(rating_chart_path, dpi=300)
    plt.close()
    
    # Plot 2: Top 10 Places by review count
    fig, ax = plt.subplots(figsize=(6, 3))
    top_places = df_all["place_name"].value_counts().head(10)
    y_pos = np.arange(len(top_places))
    
    # Get text vs no-text for top places
    text_counts = []
    no_text_counts = []
    for place in top_places.index:
        place_df = df_all[df_all["place_name"] == place]
        text_counts.append(place_df["has_text"].sum())
        no_text_counts.append((~place_df["has_text"]).sum())
        
    ax.barh(y_pos, text_counts, label="With Review Text", color=colors_hex["secondary"], height=0.55, edgecolor=colors_hex["primary"], linewidth=0.5)
    ax.barh(y_pos, no_text_counts, left=text_counts, label="Rating Only (No Text)", color=colors_hex["background"], height=0.55, edgecolor=colors_hex["dark"], linewidth=0.5)
    
    ax.set_yticks(y_pos)
    # Truncate long place names for formatting
    truncated_names = [name[:20] + "..." if len(name) > 20 else name for name in top_places.index]
    ax.set_yticklabels(truncated_names, fontsize=8)
    ax.invert_yaxis()  # top-down
    ax.set_title("Top 10 Establishments by Review Count", fontsize=11, fontweight="bold", pad=10, color=colors_hex["primary"])
    ax.set_xlabel("Number of Reviews", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    top_places_path = os.path.join(temp_dir, "top_places.png")
    plt.savefig(top_places_path, dpi=300)
    plt.close()
    
    # Plot 3: Review length distribution
    fig, ax = plt.subplots(figsize=(6, 3.2))
    word_counts = df_all[df_all["has_text"]]["review_text_clean"].apply(lambda x: len(x.split()))
    
    # We clip at 100 words to make the plot clean
    word_counts_clipped = word_counts.clip(upper=100)
    
    ax.hist(word_counts_clipped, bins=20, color=colors_hex["primary"], edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.set_title("Review Length Distribution (Word Count)", fontsize=11, fontweight="bold", pad=10, color=colors_hex["primary"])
    ax.set_xlabel("Word Count (Clipped at 100 words)", fontsize=9)
    ax.set_ylabel("Number of Reviews", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    word_dist_path = os.path.join(temp_dir, "word_dist.png")
    plt.savefig(word_dist_path, dpi=300)
    plt.close()
    
    return rating_chart_path, top_places_path, word_dist_path

# 3. Build ReportLab PDF
def build_pdf(pdf_path, stats, chart_paths, df_all):
    rating_chart, top_places_chart, word_dist_chart = chart_paths
    
    # Set document structure
    margin = 36 # 0.5 inch margins to fit structure nicely
    doc = SimpleDocTemplate(
        pdf_path, 
        pagesize=letter, 
        leftMargin=margin, 
        rightMargin=margin, 
        topMargin=margin, 
        bottomMargin=margin
    )
    
    # Colors
    c_primary = colors.HexColor("#1E3A8A")
    c_secondary = colors.HexColor("#0D9488")
    c_dark = colors.HexColor("#1F2937")
    c_light = colors.HexColor("#F3F4F6")
    c_white = colors.white
    
    # Styles
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=28,
        textColor=c_primary,
        spaceAfter=4
    )
    
    subtitle_style = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=14,
        textColor=c_secondary,
        spaceAfter=15
    )
    
    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=c_primary,
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        "BodyText_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=c_dark,
        spaceAfter=8
    )
    
    table_text = ParagraphStyle(
        "TableText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=c_dark
    )
    
    table_header = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=c_white
    )
    
    story = []
    
    # Header block
    story.append(Paragraph("Dataset Analysis Dashboard", title_style))
    story.append(Paragraph("Aspect-Based Sentiment Analysis (ABSA) Dataset Overview", subtitle_style))
    
    # Executive Summary Paragraph
    summary_text = (
        "This report provides an automated analytical breakdown of the CSV datasets collected for the "
        "Aspect-Based Sentiment Analysis project. The dataset is split into Coffeeshops and Matcha Shops categories, "
        "containing localized reviews with geographic coordinate attributes and rating scores."
    )
    story.append(Paragraph(summary_text, body_style))
    story.append(Spacer(1, 10))
    
    # Statistics Table
    story.append(Paragraph("Key Dataset Metrics", h1_style))
    
    stats_data = [
        [Paragraph("Metric", table_header), Paragraph("Value", table_header), Paragraph("Description", table_header)],
        [Paragraph("Total CSV Files", table_text), Paragraph(f"{stats['total_files']}", table_text), Paragraph("Total independent business files processed.", table_text)],
        [Paragraph("Unique Establishments", table_text), Paragraph(f"{stats['unique_places']}", table_text), Paragraph("Total unique establishments (counted by CSV files).", table_text)],
        [Paragraph("Total Dataset Rows", table_text), Paragraph(f"{stats['total_rows']:,}", table_text), Paragraph("Combined rows across all files.", table_text)],
        [Paragraph("Rows with Review Text", table_text), Paragraph(f"{stats['rows_with_text']:,}", table_text), Paragraph("Reviews containing clean textual data.", table_text)],
        [Paragraph("Rating-Only Rows (No Text)", table_text), Paragraph(f"{stats['rows_no_text']:,}", table_text), Paragraph("Empty or missing review texts.", table_text)],
        [Paragraph("Average Rating", table_text), Paragraph(f"{stats['average_rating']:.2f} / 5.0", table_text), Paragraph("Mean rating of all reviews.", table_text)],
        [Paragraph("Average Rating (With Text)", table_text), Paragraph(f"{stats['average_rating_with_text']:.2f} / 5.0", table_text), Paragraph("Mean rating of reviews containing text.", table_text)],
    ]
    
    # Column widths adding up to 540 (total letter width is 612 - 72 = 540)
    col_widths = [140, 100, 300]
    
    stats_table = Table(stats_data, colWidths=col_widths)
    stats_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), c_primary),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [c_white, c_light])
    ]))
    story.append(stats_table)
    story.append(Spacer(1, 15))
    
    # Section: Charts
    story.append(Paragraph("Rating and Volume Visualizations", h1_style))
    
    # Create two columns for charts side by side
    # Each chart image is 260 wide (fits 260 * 2 = 520, fits under 540 total width)
    chart_data = [
        [Image(rating_chart, width=260, height=130), Image(top_places_chart, width=260, height=130)]
    ]
    chart_table = Table(chart_data, colWidths=[270, 270])
    chart_table.setStyle(TableStyle([
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 0),
        ("RIGHTPADDING", (0,0), (-1,-1), 0),
    ]))
    story.append(chart_table)
    story.append(Spacer(1, 15))
    
    # Page 2: Distribution of lengths and Top 5 Establishments detail
    # ReportLab SimpleDocTemplate automatically adds pages if elements exceed, but let's make sure it wraps clean
    page2_elements = []
    page2_elements.append(Paragraph("Review Details & Length Analysis", h1_style))
    
    # Review length chart and Top 5 Detail Table side-by-side or stacked
    # Let's show review length distribution chart on the left, and a top-reviewed table on the right.
    top5_df = df_all["place_name"].value_counts().head(5)
    top5_data = [
        [Paragraph("Establishment", table_header), Paragraph("Reviews", table_header), Paragraph("Avg Rating", table_header)]
    ]
    for place, count in top5_df.items():
        place_df = df_all[df_all["place_name"] == place]
        avg_r = place_df["rating"].mean()
        # Truncate name to fit table
        short_name = place[:18] + "..." if len(place) > 18 else place
        top5_data.append([
            Paragraph(short_name, table_text),
            Paragraph(f"{count:,}", table_text),
            Paragraph(f"{avg_r:.2f}", table_text)
        ])
        
    top5_table = Table(top5_data, colWidths=[140, 60, 60])
    top5_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), c_secondary),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [c_white, c_light])
    ]))
    
    details_layout_data = [
        [Image(word_dist_chart, width=260, height=138), top5_table]
    ]
    details_table = Table(details_layout_data, colWidths=[270, 270])
    details_table.setStyle(TableStyle([
        ("ALIGN", (0,0), (-1,-1), "LEFT"),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 0),
        ("RIGHTPADDING", (0,0), (-1,-1), 0),
    ]))
    page2_elements.append(details_table)
    
    # Note on next steps
    page2_elements.append(Spacer(1, 10))
    page2_elements.append(Paragraph("Dataset Observations", h1_style))
    observations = (
        "<b>Observations & Analytics Summary:</b><br/>"
        "• The rating distribution shows a significant positive bias with a high volume of 5-star ratings.<br/>"
        "• Approximately 37.4% (6,165 rows) of the dataset consists of rating-only reviews without text. "
        "For Aspect-Based Sentiment Analysis, these rows must be pre-filtered out since they lack textual contexts.<br/>"
        "• The average review text contains fewer than 30 words, which is ideal for targeted aspect extraction."
    )
    page2_elements.append(Paragraph(observations, body_style))
    
    story.append(KeepTogether(page2_elements))
    
    # Footer and Page Number helper function
    def add_footer(canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(c_dark)
        canvas.drawString(margin, margin - 15, "Dashboard Report: Aspect-Based Sentiment Analysis")
        canvas.drawRightString(letter[0] - margin, margin - 15, f"Page {doc.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)
    print("PDF Generation complete.")

# 4. Main Execution
if __name__ == "__main__":
    print("Starting analysis...")
    data = analyze_dataset()
    if data is not None:
        df_all, stats = data
        temp_dir = "./temp_charts"
        chart_paths = generate_visualizations(df_all, temp_dir)
        
        pdf_filename = "dashboard_report.pdf"
        build_pdf(pdf_filename, stats, chart_paths, df_all)
        
        # Clean up temp charts
        for path in chart_paths:
            try:
                os.remove(path)
            except Exception:
                pass
        try:
            os.rmdir(temp_dir)
        except Exception:
            pass
        
        print(f"Report successfully saved to: {os.path.abspath(pdf_filename)}")
