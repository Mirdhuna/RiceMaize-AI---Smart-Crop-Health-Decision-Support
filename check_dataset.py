from pathlib import Path
from PIL import Image
import hashlib

root = Path("Dataset")
extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

total = 0
corrupted = []
duplicates = {}
sizes = {}

for path in root.rglob("*"):
    if not path.is_file() or path.suffix.lower() not in extensions:
        continue

    total += 1

    try:
        with Image.open(path) as image:
            image.verify()

        with Image.open(path) as image:
            size = image.size

        sizes[size] = sizes.get(size, 0) + 1

        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        duplicates.setdefault(digest, []).append(str(path))

    except Exception:
        corrupted.append(str(path))

duplicate_groups = [
    paths for paths in duplicates.values() if len(paths) > 1
]

print("Total images:", total)
print("Unique image files:", len(duplicates))
print("Corrupted images:", len(corrupted))
print("Exact duplicate groups:", len(duplicate_groups))
print("\nMost common image dimensions:")

for size, count in sorted(
    sizes.items(), key=lambda item: item[1], reverse=True
)[:10]:
    print(size, count)

if corrupted:
    print("\nCorrupted files:")
    for path in corrupted:
        print(path)

if duplicate_groups:
    print("\nDuplicate groups:")
    for group in duplicate_groups:
        print(group)