#!/usr/bin/env python3
import os
import csv
import re
import difflib
from pathlib import Path

# Paths to the datasets
BASE_DIR = Path(__file__).resolve().parent.parent
COFFEE_DIR = BASE_DIR / "Dataset" / "Coffeeshops"
MATCHA_DIR = BASE_DIR / "Dataset" / "Matcha Shops"

# Words to ignore when doing core-name comparison
NOISE_WORDS = {
    "coffee", "matcha", "cafe", "café", "shop", "shops", "bar", "davao", "dvo",
    "co", "and", "de", "the", "by", "branch", "at", "resto", "bakery", "roastery",
    "roaster", "station", "house", "dvo", "ph"
}

def clean_filename_to_name(filename: str) -> str:
    """Fallback to convert filename to a readable shop name if CSV is unreadable."""
    name = filename
    if name.endswith(".csv"):
        name = name[:-4]
    if name.endswith("_reviews"):
        name = name[:-8]
    # Replace underscores and dashes with spaces
    name = name.replace("_", " ").replace("-", " ")
    return " ".join(name.split())

def get_shop_name_from_csv(path: Path) -> str:
    """Reads the place_name column from the CSV file. Falls back to filename if needed."""
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames and "place_name" in reader.fieldnames:
                for row in reader:
                    val = row.get("place_name")
                    if val and val.strip():
                        return val.strip()
    except Exception:
        pass
    return clean_filename_to_name(path.name)

def normalize_name(name: str) -> str:
    """Lowercases, removes special characters, and collapses whitespace."""
    n = name.lower()
    # Replace common symbols with words or spaces
    n = n.replace("&", "and").replace("@", "at")
    # Remove any non-alphanumeric/non-space character
    n = re.sub(r'[^a-z0-9\s]', ' ', n)
    return " ".join(n.split())

def get_core_name(normalized_name: str) -> str:
    """Removes noise words to extract the core brand name."""
    words = normalized_name.split()
    core = [w for w in words if w not in NOISE_WORDS]
    return " ".join(core) if core else normalized_name

def scan_directory(directory: Path) -> list[dict]:
    """Scans directory for CSV files and extracts name metrics."""
    shops = []
    if not directory.exists():
        return shops
    for p in directory.glob("*.csv"):
        orig_name = get_shop_name_from_csv(p)
        norm_name = normalize_name(orig_name)
        core_name = get_core_name(norm_name)
        shops.append({
            "file_name": p.name,
            "relative_path": p.relative_to(BASE_DIR),
            "original_name": orig_name,
            "normalized_name": norm_name,
            "core_name": core_name
        })
    return sorted(shops, key=lambda x: x["original_name"])

def main():
    print("=" * 60)
    print("  Matcha vs Coffee Shop Name Duplicate & Similarity Detector  ")
    print("=" * 60)

    if not COFFEE_DIR.exists() or not MATCHA_DIR.exists():
        print(f"Error: Missing directories.")
        print(f"Expected: {COFFEE_DIR}")
        print(f"Expected: {MATCHA_DIR}")
        return

    print("Scanning datasets...")
    coffee_shops = scan_directory(COFFEE_DIR)
    matcha_shops = scan_directory(MATCHA_DIR)

    print(f"Found {len(coffee_shops)} coffee shops and {len(matcha_shops)} matcha shops.")

    exact_matches = []
    similar_matches = []
    core_matches = []

    similarity_threshold = 0.80

    # Track matches to avoid duplicate entries in list
    matched_pairs = set()

    for m in matcha_shops:
        m_norm = m["normalized_name"]
        m_core = m["core_name"]
        
        for c in coffee_shops:
            c_norm = c["normalized_name"]
            c_core = c["core_name"]

            pair_key = (m["file_name"], c["file_name"])

            # 1. Exact Match
            if m_norm == c_norm:
                exact_matches.append((m, c))
                matched_pairs.add(pair_key)
                continue

            # 2. Core Name Match (if core name is not empty or too generic)
            if m_core == c_core and len(m_core) >= 3:
                core_matches.append((m, c))
                matched_pairs.add(pair_key)
                continue

            # 3. Sequence Similarity on Normalized Name
            ratio = difflib.SequenceMatcher(None, m_norm, c_norm).ratio()
            if ratio >= similarity_threshold:
                similar_matches.append((m, c, ratio))
                matched_pairs.add(pair_key)

    # Sort similar matches by ratio descending
    similar_matches.sort(key=lambda x: x[2], reverse=True)

    # Output Results to Console
    print("\n" + "=" * 60)
    print(f"1. EXACT MATCHES ({len(exact_matches)})")
    print("=" * 60)
    for m, c in exact_matches:
        print(f"Matcha File: {m['file_name']}")
        print(f"Coffee File: {c['file_name']}")
        print(f"Place Name:  '{m['original_name']}'")
        print("-" * 50)

    print("\n" + "=" * 60)
    print(f"2. CATEGORY / CORE BRAND MATCHES ({len(core_matches)})")
    print("=" * 60)
    for m, c in core_matches:
        print(f"Matcha: '{m['original_name']}' ({m['file_name']})")
        print(f"Coffee: '{c['original_name']}' ({c['file_name']})")
        print(f"Core Brand:  '{m['core_name']}'")
        print("-" * 50)

    print("\n" + "=" * 60)
    print(f"3. HIGHLY SIMILAR NAMES (Threshold >= {similarity_threshold:.0%}) ({len(similar_matches)})")
    print("=" * 60)
    for m, c, ratio in similar_matches:
        print(f"Matcha: '{m['original_name']}' ({m['file_name']})")
        print(f"Coffee: '{c['original_name']}' ({c['file_name']})")
        print(f"Similarity:  {ratio:.1%}")
        print("-" * 50)

    # Generate a Text Report
    report_path = BASE_DIR / "Thesis-datacollectionscripts" / "duplicate_detection_report.txt"
    try:
        # Clean up old markdown report if it exists
        old_md_path = BASE_DIR / "Thesis-datacollectionscripts" / "duplicate_detection_report.md"
        if old_md_path.exists():
            old_md_path.unlink()

        with report_path.open("w", encoding="utf-8") as rep:
            rep.write("=" * 60 + "\n")
            rep.write("  Matcha vs Coffee Shop Duplicate Name Report\n")
            rep.write("=" * 60 + "\n")
            rep.write(f"Scanned {len(matcha_shops)} Matcha shops and {len(coffee_shops)} Coffee shops.\n\n")
            
            rep.write(f"1. EXACT MATCHES ({len(exact_matches)})\n")
            rep.write("-" * 60 + "\n")
            if exact_matches:
                for m, c in exact_matches:
                    rep.write(f"Matcha: '{m['original_name']}' ({m['file_name']})\n")
                    rep.write(f"Coffee: '{c['original_name']}' ({c['file_name']})\n")
                    rep.write("-" * 40 + "\n")
            else:
                rep.write("No exact matches found.\n")
            rep.write("\n")

            rep.write(f"2. CATEGORY / CORE BRAND MATCHES ({len(core_matches)})\n")
            rep.write("-" * 60 + "\n")
            if core_matches:
                for m, c in core_matches:
                    rep.write(f"Matcha: '{m['original_name']}' ({m['file_name']})\n")
                    rep.write(f"Coffee: '{c['original_name']}' ({c['file_name']})\n")
                    rep.write(f"Core Brand: '{m['core_name']}'\n")
                    rep.write("-" * 40 + "\n")
            else:
                rep.write("No core brand matches found.\n")
            rep.write("\n")

            rep.write(f"3. HIGHLY SIMILAR NAMES (Threshold >= {similarity_threshold:.0%}) ({len(similar_matches)})\n")
            rep.write("-" * 60 + "\n")
            if similar_matches:
                for m, c, ratio in similar_matches:
                    rep.write(f"Matcha: '{m['original_name']}' ({m['file_name']})\n")
                    rep.write(f"Coffee: '{c['original_name']}' ({c['file_name']})\n")
                    rep.write(f"Similarity: {ratio:.1%}\n")
                    rep.write("-" * 40 + "\n")
            else:
                rep.write("No similar matches found.\n")
            
        print(f"\nText report generated successfully at: {report_path}")
    except Exception as e:
        print(f"Error generating report: {e}")

if __name__ == "__main__":
    main()
