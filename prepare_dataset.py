
from pathlib import Path
import hashlib
import csv

ROOT = Path("Dataset")
OUTPUT = Path("data")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

OUTPUT.mkdir(exist_ok=True)

seen = {}
duplicates = []
records = []

def file_hash(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()

for path in sorted(ROOT.rglob("*")):
    if not path.is_file():
        continue

    if path.suffix.lower() not in EXTENSIONS:
        continue

    relative = path.relative_to(ROOT)
    parts = relative.parts

    if len(parts) < 2:
        continue

    crop = parts[0]
    category = parts[1]

    if category == "Healthy":
        label = f"{crop}/Healthy"
    elif len(parts) >= 3:
        label = f"{crop}/{category}/{parts[2]}"
    else:
        continue

    digest = file_hash(path)

    if digest in seen:
        duplicates.append({
            "duplicate": str(path),
            "original": seen[digest],
            "label": label
        })
        continue

    seen[digest] = str(path)

    records.append({
        "image_path": str(path),
        "crop": crop,
        "category": category,
        "label": label
    })

with (OUTPUT / "image_manifest.csv").open(
    "w", newline="", encoding="utf-8"
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["image_path", "crop", "category", "label"]
    )
    writer.writeheader()
    writer.writerows(records)

with (OUTPUT / "duplicates.csv").open(
    "w", newline="", encoding="utf-8"
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["duplicate", "original", "label"]
    )
    writer.writeheader()
    writer.writerows(duplicates)

print("Original image files:", len(records) + len(duplicates))
print("Unique images:", len(records))
print("Duplicate copies excluded:", len(duplicates))
print("Classes:", len(set(row["label"] for row in records)))
print("\nImages per class:")

counts = {}

for row in records:
    counts[row["label"]] = counts.get(row["label"], 0) + 1

for label, count in sorted(counts.items()):
    print(f"{label}: {count}")

print("\nSaved data/image_manifest.csv")
print("Saved data/duplicates.csv")
