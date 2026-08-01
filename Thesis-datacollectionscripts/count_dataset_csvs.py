from pathlib import Path
import csv
import hashlib


DATASET_DIR = Path("Dataset/Coffeeshops")


def file_hash(path: Path) -> str:
    # Return the SHA-256 hash of a file.
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def count_instances(path: Path) -> int:
    #Count each non-empty row as one instance, without counting feature values separately.
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            return 0

        instance_count = 0
        for row in reader:
            if not row:
                continue
            if all((value or "").strip() == "" for value in row.values()):
                continue
            instance_count += 1

    return instance_count


def main() -> None:
    if not DATASET_DIR.exists():
        print(f"Dataset folder not found: {DATASET_DIR}")
        return

    csv_files = sorted(DATASET_DIR.glob("*.csv"))
    print(f"Total CSV files in {DATASET_DIR}: {len(csv_files)}")

    total_instances = sum(count_instances(csv_file) for csv_file in csv_files)
    print(f"Total instances across CSV files (excluding features): {total_instances}")

    hashes: dict[str, list[Path]] = {}
    for csv_file in csv_files:
        digest = file_hash(csv_file)
        hashes.setdefault(digest, []).append(csv_file)

    duplicate_groups = [files for files in hashes.values() if len(files) > 1]
    duplicate_file_count = sum(len(files) - 1 for files in duplicate_groups)

    print(f"Duplicate file groups: {len(duplicate_groups)}")
    print(f"Duplicate files beyond originals: {duplicate_file_count}")

    if duplicate_groups:
        print("\nDuplicate CSV files:")
        for group_number, files in enumerate(duplicate_groups, start=1):
            print(f"\nGroup {group_number} ({len(files)} identical files):")
            for file_path in files:
                print(f"  - {file_path}")
    else:
        print("\nNo duplicate CSV files found.")


main()
