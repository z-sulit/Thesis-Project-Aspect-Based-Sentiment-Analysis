from pathlib import Path
import csv
import hashlib
import re
from typing import Dict, List, Set, Tuple

# Define root directory relative to the script location
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "Thesis-datacollectionscripts" else SCRIPT_DIR

MATCHA_DIR = ROOT_DIR / "Dataset/Matcha Shops"
# Fallback to the specific casing from the request if the standard one doesn't exist
if not MATCHA_DIR.exists():
    MATCHA_DIR = ROOT_DIR / "Dataset/Matcha SHops"

COFFEE_DIR = ROOT_DIR / "Dataset/Coffeeshops"


def get_file_hash(path: Path) -> str:

    """Return the SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    try:
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return ""


def normalize_text(text: str) -> str:
    """Normalize text by converting to lowercase and removing non-alphanumeric characters."""
    if not text:
        return ""
    # Strip common ellipsis symbols
    t = text.lower().strip()
    while t.endswith("…") or t.endswith("...") or t.endswith(".."):
        t = t.rstrip(".…").strip()
    while t.startswith("…") or t.startswith("...") or t.startswith(".."):
        t = t.lstrip(".…").strip()
    
    # Remove all non-alphanumeric characters
    return re.sub(r'[^a-z0-9]', '', t)


def read_csv_reviews(path: Path) -> List[Dict[str, str]]:
    """Read a CSV file and return a list of rows."""
    rows = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return []
            for row in reader:
                if row:
                    rows.append(row)
    except Exception as e:
        print(f"Error reading {path.name}: {e}")
    return rows


def main() -> None:
    print("=" * 60)
    print("Dataset Duplicate Finder: Matcha Shops vs Coffeeshops")
    print("=" * 60)

    # Validate paths
    if not MATCHA_DIR.exists():
        print(f"Error: Matcha directory '{MATCHA_DIR}' not found.")
        return
    if not COFFEE_DIR.exists():
        print(f"Error: Coffeeshops directory '{COFFEE_DIR}' not found.")
        return

    matcha_files = sorted(MATCHA_DIR.glob("*.csv"))
    coffee_files = sorted(COFFEE_DIR.glob("*.csv"))

    print(f"Found {len(matcha_files)} CSV files in Matcha dataset: {MATCHA_DIR}")
    print(f"Found {len(coffee_files)} CSV files in Coffeeshops dataset: {COFFEE_DIR}")
    print("-" * 60)

    # 1. Filename comparisons (case-insensitive, ignoring extension)
    coffee_names = {f.name.lower(): f for f in coffee_files}
    matcha_names = {f.name.lower(): f for f in matcha_files}

    filename_matches: List[Tuple[Path, Path]] = []
    for m_file in matcha_files:
        m_name_lower = m_file.name.lower()
        if m_name_lower in coffee_names:
            filename_matches.append((m_file, coffee_names[m_name_lower]))

    print(f"1. Filename Matches: {len(filename_matches)} files have the same name in both folders.")

    # 2. File hash comparisons (exact identical copies)
    coffee_hashes = {get_file_hash(f): f for f in coffee_files if get_file_hash(f)}
    hash_matches: List[Tuple[Path, Path]] = []
    for m_file in matcha_files:
        m_hash = get_file_hash(m_file)
        if m_hash and m_hash in coffee_hashes:
            hash_matches.append((m_file, coffee_hashes[m_hash]))

    print(f"2. Exact Content Hash Matches: {len(hash_matches)} files are 100% byte-for-byte identical.")

    # 3. Row-level duplicate/overlap analysis
    print("3. Performing row-level comparison...")
    
    # Index Coffeeshop reviews by their normalized prefix
    # prefix (30 chars) -> list of (coffee_file_path, normalized_full_text)
    coffee_index: Dict[str, List[Tuple[Path, str]]] = {}
    coffee_file_totals: Dict[Path, int] = {}
    
    for c_file in coffee_files:
        rows = read_csv_reviews(c_file)
        valid_reviews_count = 0
        for row in rows:
            review_text = row.get("review_text") or row.get("review") or ""
            norm = normalize_text(review_text)
            if not norm:
                continue
            valid_reviews_count += 1
            # Index by the first 30 characters
            pref = norm[:30]
            coffee_index.setdefault(pref, []).append((c_file, norm))
        coffee_file_totals[c_file] = valid_reviews_count

    # Compare Matcha reviews against index
    # matcha_file -> {coffee_file -> matching_review_count}
    overlap_results: Dict[Path, Dict[Path, int]] = {}
    matcha_file_totals: Dict[Path, int] = {}

    for m_file in matcha_files:
        rows = read_csv_reviews(m_file)
        valid_reviews_count = 0
        matched_reviews: Set[Tuple[Path, str]] = set() # Avoid double-counting same match
        
        for row in rows:
            review_text = row.get("review_text") or row.get("review") or ""
            norm = normalize_text(review_text)
            if not norm:
                continue
            valid_reviews_count += 1
            
            # Find candidate matches using the first 30 characters prefix
            pref = norm[:30]
            candidates = coffee_index.get(pref, [])
            
            # Check substrings to account for truncation
            for c_file, c_norm in candidates:
                if len(norm) >= 15 and len(c_norm) >= 15:
                    # If both are substantial, check substring overlap (due to truncation)
                    if norm in c_norm or c_norm in norm:
                        matched_reviews.add((c_file, norm))
                elif norm == c_norm:
                    # For short texts, require exact match
                    matched_reviews.add((c_file, norm))
                    
        matcha_file_totals[m_file] = valid_reviews_count
        
        # Aggregate matches for this Matcha file
        for c_file, _ in matched_reviews:
            overlap_results.setdefault(m_file, {})
            overlap_results[m_file][c_file] = overlap_results[m_file].get(c_file, 0) + 1

    # Identify duplicate files based on row overlap percentage
    # We define a "duplicate" if:
    # - Overlap is >= 50% of the reviews in the Matcha file, and
    # - At least 1 review matches (or 0 if both files have 0 reviews but matching filenames)
    high_overlap_pairs: List[Dict] = []
    
    for m_file, c_matches in overlap_results.items():
        m_total = matcha_file_totals.get(m_file, 0)
        for c_file, match_count in c_matches.items():
            c_total = coffee_file_totals.get(c_file, 0)
            overlap_pct_matcha = (match_count / m_total * 100) if m_total > 0 else 0
            overlap_pct_coffee = (match_count / c_total * 100) if c_total > 0 else 0
            
            if overlap_pct_matcha >= 50.0:
                high_overlap_pairs.append({
                    "matcha_file": m_file,
                    "coffee_file": c_file,
                    "match_count": match_count,
                    "matcha_total": m_total,
                    "coffee_total": c_total,
                    "matcha_overlap_pct": overlap_pct_matcha,
                    "coffee_overlap_pct": overlap_pct_coffee
                })

    # Add cases where files have same name but 0 reviews (e.g. empty shops)
    for m_file, c_file in filename_matches:
        m_total = matcha_file_totals.get(m_file, 0)
        c_total = coffee_file_totals.get(c_file, 0)
        # Check if already added
        already_added = any(
            p["matcha_file"] == m_file and p["coffee_file"] == c_file 
            for p in high_overlap_pairs
        )
        if not already_added and m_total == 0 and c_total == 0:
            high_overlap_pairs.append({
                "matcha_file": m_file,
                "coffee_file": c_file,
                "match_count": 0,
                "matcha_total": 0,
                "coffee_total": 0,
                "matcha_overlap_pct": 100.0,
                "coffee_overlap_pct": 100.0
            })

    # Sort high overlap pairs by overlap percentage descending
    high_overlap_pairs.sort(key=lambda x: x["matcha_overlap_pct"], reverse=True)

    print(f"4. Row-level Duplicates (>=50% review overlap): {len(high_overlap_pairs)} pairs identified.")
    print("-" * 60)

    # Print summary of matches
    if high_overlap_pairs:
        print(f"{'Matcha File':<40} {'Coffee File':<40} {'Overlap %':<10} {'Overlap Count':<15}")
        print("-" * 110)
        for pair in high_overlap_pairs[:20]: # Print top 20
            m_name = pair["matcha_file"].name
            c_name = pair["coffee_file"].name
            pct = f"{pair['matcha_overlap_pct']:.1f}%"
            cnt = f"{pair['match_count']}/{pair['matcha_total']}"
            print(f"{m_name:<40} {c_name:<40} {pct:<10} {cnt:<15}")
        if len(high_overlap_pairs) > 20:
            print(f"... and {len(high_overlap_pairs) - 20} more pairs.")
    else:
        print("No duplicate/high-overlap pairs found.")

    # Generate Reports
    txt_report_path = ROOT_DIR / "Dataset/duplicate_report.txt"
    generate_txt_report(txt_report_path, filename_matches, hash_matches, high_overlap_pairs)
    print("-" * 60)
    print(f"Detailed report saved to:")
    print(f"  Text: {txt_report_path.resolve()}")
    print("=" * 60)


def generate_txt_report(
    report_path: Path, 
    filename_matches: List[Tuple[Path, Path]], 
    hash_matches: List[Tuple[Path, Path]], 
    high_overlap_pairs: List[Dict]
) -> None:
    """Generate a clean plain-text report of the duplicates."""
    lines = [
        "=" * 120,
        "DATASET COMPARISON REPORT: MATCHA SHOPS VS COFFEESHOPS",
        "=" * 120,
        "",
        "This report identifies duplicate CSV files and overlapping reviews between the Matcha Shops and Coffeeshops datasets.",
        "",
        "SUMMARY METRICS",
        "---------------",
        f"- Filename Matches: {len(filename_matches)} files have matching names across both directories.",
        f"- Exact Byte-for-Byte Matches (SHA256): {len(hash_matches)} files are completely identical.",
        f"- Semantic Duplicates (>=50% Review Overlap): {len(high_overlap_pairs)} pairs of files have significant review overlap.",
        "",
        "DETAILED OVERLAP ANALYSIS (>=50% Review Overlap)",
        "------------------------------------------------",
        "Note: Overlap % is relative to the number of reviews in the Matcha file.",
        "",
        f"{'Matcha File':<45} {'Coffeeshop File':<45} {'Overlap % (Matcha)':<20} {'Overlap % (Coffee)':<20} {'Match Count':<12} {'Matcha Total':<13} {'Coffeeshop Total':<16}",
        "-" * 174
    ]

    for pair in high_overlap_pairs:
        m_name = pair["matcha_file"].name
        c_name = pair["coffee_file"].name
        pct_m = f"{pair['matcha_overlap_pct']:.1f}%"
        pct_c = f"{pair['coffee_overlap_pct']:.1f}%"
        cnt = str(pair["match_count"])
        m_tot = str(pair["matcha_total"])
        c_tot = str(pair["coffee_total"])
        lines.append(f"{m_name:<45} {c_name:<45} {pct_m:<20} {pct_c:<20} {cnt:<12} {m_tot:<13} {c_tot:<16}")

    lines.extend([
        "",
        "EXACT FILENAME MATCHES",
        "----------------------",
        "Files present in both folders with matching filenames:",
        ""
    ])

    if filename_matches:
        for m_file, c_file in filename_matches:
            lines.append(f"- {m_file.name}")
    else:
        lines.append("None found.")

    try:
        # Create directory if it doesn't exist
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(lines), encoding="utf-8")
    except Exception as e:
        print(f"Failed to write report file: {e}")


if __name__ == "__main__":
    main()


