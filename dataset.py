from pathlib import Path
import csv
import random

from PIL import Image, ImageEnhance, ImageOps

IMAGE_SIZE = 224
SPLITS_PATH = Path("data/splits.csv")
CLASS_MAP_PATH = Path("data/class_map.csv")


def load_split_rows(split, splits_path=SPLITS_PATH):
    with Path(splits_path).open(newline="", encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file) if row["split"] == split]

    if not rows:
        raise ValueError(f"No rows found for split={split!r} in {splits_path}")

    return rows


def load_class_names(class_map_path=CLASS_MAP_PATH):
    with Path(class_map_path).open(newline="", encoding="utf-8") as file:
        rows = sorted(csv.DictReader(file), key=lambda row: int(row["class_id"]))

    return [row["label"] for row in rows]


def _resize_rgb(image, size=IMAGE_SIZE):
    image = image.convert("RGB")
    resample = getattr(Image, "Resampling", Image).BILINEAR
    return image.resize((size, size), resample)


def _augment(image, rng):
    if rng.random() < 0.5:
        image = ImageOps.mirror(image)

    angle = rng.uniform(-15, 15)
    image = image.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))

    if rng.random() < 0.8:
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.8, 1.2))
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.8, 1.2))
        image = ImageEnhance.Color(image).enhance(rng.uniform(0.8, 1.2))

    return image


def preprocess_image(path, augment=False, image_size=IMAGE_SIZE, rng=None):
    with Image.open(path) as image:
        image = _resize_rgb(image, image_size)

        if augment:
            image = _augment(image, rng or random.Random())

        pixels = list(image.getdata())

    scale = 1.0 / 255.0
    channels = [[], [], []]

    for red, green, blue in pixels:
        channels[0].append(red * scale)
        channels[1].append(green * scale)
        channels[2].append(blue * scale)

    return channels


class CropImageDataset:
    """Loads original images. Resize, RGB, and 0-1 scaling happen here."""

    def __init__(self, split, augment=None, image_size=IMAGE_SIZE, seed=42):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")

        self.split = split
        self.augment = (split == "train") if augment is None else augment
        self.image_size = image_size
        self.rng = random.Random(seed)
        self.rows = load_split_rows(split)
        self.class_names = load_class_names()

        if self.augment and split != "train":
            raise ValueError("Augmentation is only allowed for the training split.")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        channels = preprocess_image(
            row["image_path"],
            augment=self.augment,
            image_size=self.image_size,
            rng=self.rng,
        )
        return channels, int(row["class_id"])


def iter_batches(dataset, batch_size=32, shuffle=False, seed=42):
    indices = list(range(len(dataset)))

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        images = []
        labels = []

        for index in batch_indices:
            image, label = dataset[index]
            images.append(image)
            labels.append(label)

        yield images, labels


if __name__ == "__main__":
    train = CropImageDataset("train")
    val = CropImageDataset("val")
    test = CropImageDataset("test")

    image, label = train[0]
    print("Train images:", len(train))
    print("Val images:", len(val))
    print("Test images:", len(test))
    print("Classes:", len(train.class_names))
    print("Sample channels x pixels:", len(image), len(image[0]))
    print("Sample label:", label, train.class_names[label])
    print("Pixel range:", min(image[0]), max(image[0]))
    print("Augment train/val/test:", train.augment, val.augment, test.augment)
