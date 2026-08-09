import os
import re
import sys
import time
from pathlib import Path
import pandas as pd
import ftfy

# Fix terminal display encoding on Windows for emojis log
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def fix_mojibake_and_symbols(text):
    if pd.isna(text):
        return ""
    text = str(text)
    # Restore ₱, 🩷, —, ’, and emojis
    text = ftfy.fix_text(text)
    # Strip HTML tags but leave emojis & punctuation intact
    return re.sub(r'<[^>]+>', ' ', text).strip()


# Locate Dataset directory
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "Dataset"
if not DATASET_DIR.exists():
    DATASET_DIR = Path(__file__).resolve().parent / "Dataset"

target_folders = [DATASET_DIR / "Coffeeshops", DATASET_DIR / "Matcha Shops"]

total_files = 0
total_rows = 0

print("=" * 60)
print("STARTING IN-PLACE MOJIBAKE & EMOJI REPAIR")
print("=" * 60)

for folder in target_folders:
    if not folder.exists():
        continue

    csv_files = [f for f in folder.glob("*.csv") if not f.name.endswith(".tmp")]
    print(f"\nProcessing {len(csv_files)} files in: {folder.name}...")

    for csv_file in csv_files:
        try:
            # Read CSV with encoding fallback
            df = None
            for enc in ['utf-8', 'utf-8-sig', 'cp1252', 'latin1']:
                try:
                    df = pd.read_csv(csv_file, encoding=enc)
                    break
                except Exception:
                    continue

            if df is None:
                continue

            # Identify review_text column
            review_col = next((col for col in df.columns if col.lower() == 'review_text'), None)
            if review_col is None:
                continue

            # Repair review_text directly in-place
            df[review_col] = df[review_col].apply(fix_mojibake_and_symbols)

            # Save updated CSV with utf-8-sig
            temp_path = csv_file.with_suffix('.tmp')
            df.to_csv(temp_path, index=False, encoding='utf-8-sig')

            # Windows file replace retry logic
            replaced = False
            for _ in range(5):
                try:
                    os.replace(temp_path, csv_file)
                    replaced = True
                    break
                except PermissionError:
                    time.sleep(0.3)
            
            if not replaced and temp_path.exists():
                temp_path.unlink()

            total_files += 1
            total_rows += len(df)

        except Exception as e:
            print(f"Error repairing {csv_file.name}: {e}")

print("\n" + "=" * 60)
print(f"SUCCESS: Repaired {total_files} CSV files ({total_rows:,} review rows).")
print("=" * 60)
