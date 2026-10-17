from collections import Counter, defaultdict
from pathlib import Path
import csv
import random

from PIL import Image

MANIFEST = Path("data/image_manifest.csv")
OUTPUT = Path("data")
IMAGE_SIZE = 224
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42


def load_manifest(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def validate_image(path):
    with Image.open(path) as image:
        image.verify()

    with Image.open(path) as image:
        width, height = image.size
        mode = image.mode
        image.convert("RGB")

    return width, height, mode


def assign_class_ids(records):
    labels = sorted({row["label"] for row in records})
    class_map = {label: index for index, label in enumerate(labels)}

    for row in records:
        row["class_id"] = class_map[row["label"]]

    return class_map


def stratified_split(records, seed=SEED):
    grouped = defaultdict(list)

    for row in records:
        grouped[row["label"]].append(row)

    rng = random.Random(seed)
    split_records = []

    for label in sorted(grouped):
        items = grouped[label]
        rng.shuffle(items)

        n = len(items)
        n_test = int(round(n * TEST_RATIO))
        n_val = int(round(n * VAL_RATIO))
        n_train = n - n_val - n_test

        if n_train < 1 or n_val < 1 or n_test < 1:
            raise ValueError(
                f"Class {label} has {n} images; too few for a 70/15/15 split."
            )

        for row in items[:n_train]:
            row["split"] = "train"

        for row in items[n_train:n_train + n_val]:
            row["split"] = "val"

        for row in items[n_train + n_val:]:
            row["split"] = "test"

        split_records.extend(items)

    split_records.sort(key=lambda row: (row["split"], row["label"], row["image_path"]))
    return split_records


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    if not MANIFEST.exists():
        raise SystemExit(
            "data/image_manifest.csv not found. Run prepare_dataset.py first."
        )

    OUTPUT.mkdir(exist_ok=True)
    records = load_manifest(MANIFEST)

    if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 1e-9:
        raise SystemExit("Train/validation/test ratios must sum to 1.")

    valid = []
    unreadable = []
    modes = Counter()
    sizes = Counter()

    for row in records:
        path = Path(row["image_path"])

        try:
            width, height, mode = validate_image(path)
        except Exception as error:
            unreadable.append({"image_path": row["image_path"], "error": str(error)})
            continue

        modes[mode] += 1
        sizes[(width, height)] += 1
        valid.append(row)

    if unreadable:
        write_csv(
            OUTPUT / "unreadable_images.csv",
            unreadable,
            ["image_path", "error"],
        )
        raise SystemExit(
            f"Found {len(unreadable)} unreadable images. "
            "See data/unreadable_images.csv"
        )

    class_map = assign_class_ids(valid)
    split_records = stratified_split(valid)

    class_rows = []

    for label, class_id in class_map.items():
        sample = next(row for row in valid if row["label"] == label)
        class_rows.append({
            "class_id": class_id,
            "label": label,
            "crop": sample["crop"],
            "category": sample["category"],
        })

    write_csv(
        OUTPUT / "class_map.csv",
        class_rows,
        ["class_id", "label", "crop", "category"],
    )
    write_csv(
        OUTPUT / "splits.csv",
        [
            {
                "image_path": row["image_path"],
                "crop": row["crop"],
                "category": row["category"],
                "label": row["label"],
                "class_id": row["class_id"],
                "split": row["split"],
            }
            for row in split_records
        ],
        ["image_path", "crop", "category", "label", "class_id", "split"],
    )

    split_counts = Counter(row["split"] for row in split_records)

    print("Preprocessing plan")
    print("- Keep original images on disk")
    print(f"- Resize to {IMAGE_SIZE}x{IMAGE_SIZE}, RGB, and 0-1 scale at load time")
    print("- Augment training images only, during training")
    print()
    print("Validated unique images:", len(valid))
    print("Unreadable images:", len(unreadable))
    print("Classes:", len(class_map))
    print("Color modes:", dict(modes))
    print("Distinct original sizes:", len(sizes))
    print()
    print("Split counts:")
    for split in ("train", "val", "test"):
        print(f"  {split}: {split_counts[split]}")
    print()
    print("Per-class split:")
    print(f"{'class_id':>8}  {'train':>5}  {'val':>5}  {'test':>5}  label")

    by_class = defaultdict(Counter)

    for row in split_records:
        by_class[row["class_id"]][row["split"]] += 1

    for class_id, label in sorted((v, k) for k, v in class_map.items()):
        counts = by_class[class_id]
        print(
            f"{class_id:8d}  {counts['train']:5d}  {counts['val']:5d}  "
            f"{counts['test']:5d}  {label}"
        )

    print()
    print("Saved data/class_map.csv")
    print("Saved data/splits.csv")


if __name__ == "__main__":
    main()
