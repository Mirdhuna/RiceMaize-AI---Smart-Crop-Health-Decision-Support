from pathlib import Path
import csv
import hashlib

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from dataset import IMAGE_SIZE, SPLITS_PATH, load_class_names, load_split_rows

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ROOT_DIR = Path(__file__).resolve().parent
CACHE_DIR = ROOT_DIR / "data" / "cache_224"


def cached_image_path(source_path, image_size=IMAGE_SIZE, cache_dir=CACHE_DIR):
    digest = hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()[:16]
    stem = Path(source_path).stem
    return cache_dir / str(image_size) / "{}_{}.jpg".format(stem, digest)


def _resize_rgb(image, size):
    resample = getattr(Image, "Resampling", Image).BILINEAR
    return image.convert("RGB").resize((size, size), resample)


def ensure_resized_cache(splits_path=SPLITS_PATH, image_size=IMAGE_SIZE, cache_dir=CACHE_DIR):
    cache_dir.joinpath(str(image_size)).mkdir(parents=True, exist_ok=True)
    created = 0

    with Path(splits_path).open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    for row in rows:
        source = Path(row["image_path"])
        target = cached_image_path(source, image_size=image_size, cache_dir=cache_dir)
        if target.exists():
            continue
        with Image.open(source) as image:
            resized = _resize_rgb(image, image_size)
            resized.save(target, format="JPEG", quality=95)
        created += 1

    return created


def build_transforms(split, image_size=IMAGE_SIZE, already_resized=False):
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ops = []

    if not already_resized:
        ops.append(transforms.Resize((image_size, image_size)))

    if split == "train":
        ops.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15, fill=0),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ]
        )

    ops.extend([to_tensor, normalize])
    return transforms.Compose(ops)


def preprocess_pil(image, image_size=IMAGE_SIZE):
    image = image.convert("RGB")
    return build_transforms("test", image_size=image_size)(image)


class TorchCropDataset(Dataset):
    """Loads split rows with the same 224 RGB pipeline used at training time."""

    def __init__(self, split, augment=None, image_size=IMAGE_SIZE, use_cache=True):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")

        self.split = split
        self.augment = (split == "train") if augment is None else augment
        self.image_size = image_size
        self.use_cache = use_cache
        self.rows = load_split_rows(split)
        self.class_names = load_class_names()
        transform_split = "train" if self.augment else "val"

        if self.augment and split != "train":
            raise ValueError("Augmentation is only allowed for the training split.")

        self.transform = build_transforms(
            transform_split, image_size=image_size, already_resized=self.use_cache
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image_path = Path(row["image_path"])
        if self.use_cache:
            cached = cached_image_path(image_path, self.image_size)
            if cached.exists():
                image_path = cached

        with Image.open(image_path) as image:
            image_rgb = image.convert("RGB")
            if image_rgb.size != (self.image_size, self.image_size):
                image_rgb = _resize_rgb(image_rgb, self.image_size)
            tensor = self.transform(image_rgb)
        return tensor, int(row["class_id"])


def inference_transform(image_size=IMAGE_SIZE):
    return build_transforms("test", image_size=image_size)


def load_image_tensor(path, image_size=IMAGE_SIZE):
    with Image.open(Path(path)) as image:
        return preprocess_pil(image, image_size=image_size)
