from __future__ import annotations

import csv
import math
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

CLASS_NAMES = (
    "Enlarged Cardiomediastinum", "Pneumothorax", "Consolidation", "Pneumonia",
    "Edema", "Cardiomegaly", "Atelectasis", "Lung Opacity",
    "Pleural Effusion", "No Finding",
)
NUM_CLASSES = len(CLASS_NAMES)
VIEW_VOCAB = {"<UNK>": 0, "AP": 1, "PA": 2, "LATERAL": 3, "LL": 4, "PA LLD": 5}
ORIENTATION_VOCAB = {"<UNK>": 0, "Erect": 1, "Recumbent": 2}


@dataclass
class ImageRecord:
    path: Path
    view_id: int
    orientation_id: int
    rows: int
    columns: int


@dataclass
class StudyRecord:
    subject_id: str
    study_id: str
    images: list[ImageRecord]
    targets: Optional[torch.Tensor]


def _parse_targets(raw: str, row_number: int) -> torch.Tensor:
    target = torch.zeros(NUM_CLASSES, dtype=torch.float32)
    if not raw:
        raise ValueError(f"row {row_number}: Predict_class is empty")
    try:
        class_ids = {int(value) for value in raw.split(";")}
    except ValueError as exc:
        raise ValueError(f"row {row_number}: invalid Predict_class={raw!r}") from exc
    if not class_ids.issubset(range(1, NUM_CLASSES + 1)):
        raise ValueError(f"row {row_number}: class IDs must be in 1..10, got {class_ids}")
    for class_id in class_ids:
        target[class_id - 1] = 1.0
    if target[9] and target[:9].any():
        raise ValueError(f"row {row_number}: No Finding conflicts with disease labels")
    return target


def read_studies(csv_path, image_root, max_studies=None) -> list[StudyRecord]:
    """Stream CSV rows while retaining only a compact Study/image index."""
    csv_path, image_root = Path(csv_path), Path(image_root)
    grouped: OrderedDict[tuple[str, str], StudyRecord] = OrderedDict()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"Dicom_id", "Subject_id", "Study_id", "Rows", "Columns"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
        has_targets = "Predict_class" in (reader.fieldnames or ())
        for row_number, row in enumerate(reader, start=2):
            key = (row["Subject_id"], row["Study_id"])
            if key not in grouped:
                if max_studies is not None and len(grouped) >= max_studies:
                    break
                target = _parse_targets(row["Predict_class"], row_number) if has_targets else None
                grouped[key] = StudyRecord(key[0], key[1], [], target)
            record = grouped[key]
            if has_targets:
                target = _parse_targets(row["Predict_class"], row_number)
                if not torch.equal(record.targets, target):
                    raise ValueError(f"inconsistent labels for Study key {key}")
            path = image_root / key[0] / key[1] / f"{row['Dicom_id']}.jpg"
            if not path.is_file():
                png_path = path.with_suffix(".png")
                if not png_path.is_file():
                    raise FileNotFoundError(f"image listed in CSV does not exist: {path}")
                path = png_path
            record.images.append(ImageRecord(
                path=path,
                view_id=VIEW_VOCAB.get(row.get("ViewPosition") or "<UNK>", 0),
                orientation_id=ORIENTATION_VOCAB.get(row.get("PatientOrientation") or "<UNK>", 0),
                rows=int(row["Rows"]),
                columns=int(row["Columns"]),
            ))
    if not grouped:
        raise ValueError(f"no studies found in {csv_path}")
    return list(grouped.values())


def _letterbox_gray(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    scale = min(size / width, size / height)
    target_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    # Lanczos preserves fine lines and small opacities better when reducing large radiographs.
    resampling = Image.Resampling.LANCZOS if scale < 1.0 else Image.Resampling.BICUBIC
    resized = image.resize(target_size, resampling)
    canvas = Image.new("L", (size, size), color=128)
    canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return canvas


def _to_tensor(image: Image.Image) -> torch.Tensor:
    size = image.size[0]
    tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    return tensor.view(1, size, size).float().div_(127.5).sub_(1.0)


def progressive_augmentation_strength(epoch: int, ramp_start: int = 3,
                                      ramp_end: int = 15) -> float:
    """Return a [0, 1] curriculum strength: mild first, full after ramp_end."""
    if epoch <= ramp_start:
        return 0.0
    return min(1.0, (epoch - ramp_start) / (ramp_end - ramp_start))


def _progressive_augment(image: Image.Image, strength: float) -> Image.Image:
    """Apply conservative, label-preserving CXR geometry and exposure changes."""
    rotation = 2.0 + 5.0 * strength
    translation = 0.01 + 0.03 * strength
    scale_delta = 0.01 + 0.04 * strength
    brightness_delta = 0.02 + 0.08 * strength
    contrast_delta = 0.02 + 0.13 * strength
    gamma_delta = 0.10 * strength

    angle = math.radians(random.uniform(-rotation, rotation))
    scale = random.uniform(1.0 - scale_delta, 1.0 + scale_delta)
    width, height = image.size
    translate_x = random.uniform(-translation, translation) * width
    translate_y = random.uniform(-translation, translation) * height
    center_x, center_y = width / 2.0, height / 2.0

    # PIL expects the inverse mapping from output coordinates to input coordinates.
    inverse_scale = 1.0 / scale
    cosine = math.cos(angle) * inverse_scale
    sine = math.sin(angle) * inverse_scale
    affine = (
        cosine,
        sine,
        center_x - cosine * (center_x + translate_x) - sine * (center_y + translate_y),
        -sine,
        cosine,
        center_y + sine * (center_x + translate_x) - cosine * (center_y + translate_y),
    )
    image = ImageEnhance.Brightness(image).enhance(
        random.uniform(1.0 - brightness_delta, 1.0 + brightness_delta)
    )
    image = ImageEnhance.Contrast(image).enhance(
        random.uniform(1.0 - contrast_delta, 1.0 + contrast_delta)
    )
    if gamma_delta:
        gamma = random.uniform(1.0 - gamma_delta, 1.0 + gamma_delta)
        lookup = [round(255.0 * ((value / 255.0) ** gamma)) for value in range(256)]
        image = image.point(lookup)
    return image.transform(
        image.size, Image.Transform.AFFINE, affine,
        resample=Image.Resampling.BILINEAR, fillcolor=128,
    )


def load_multiscale_image(path: Path, global_size: int, local_canvas_size: int,
                          augment: bool, augmentation_strength: Optional[float] = None):
    if local_canvas_size % 2:
        raise ValueError("local_canvas_size must be even for a 2x2 split")
    with Image.open(path) as source:
        image = source.convert("L")
        if augment:
            if augmentation_strength is None:
                image = image.rotate(
                    random.uniform(-5.0, 5.0),
                    resample=Image.Resampling.BILINEAR, fillcolor=128,
                )
                image = ImageEnhance.Brightness(image).enhance(random.uniform(0.95, 1.05))
                image = ImageEnhance.Contrast(image).enhance(random.uniform(0.95, 1.05))
            else:
                image = _progressive_augment(image, augmentation_strength)
        global_image = _letterbox_gray(image, global_size)
        local_canvas = _letterbox_gray(image, local_canvas_size)
    patch = local_canvas_size // 2
    boxes = (
        (0, 0, patch, patch), (patch, 0, local_canvas_size, patch),
        (0, patch, patch, local_canvas_size),
        (patch, patch, local_canvas_size, local_canvas_size),
    )
    local_patches = torch.stack([_to_tensor(local_canvas.crop(box)) for box in boxes])
    return _to_tensor(global_image), local_patches


def _numeric_metadata(record: ImageRecord) -> torch.Tensor:
    """V1 numeric inputs: normalized log height, width, and aspect ratio."""
    return torch.tensor((
        math.log(max(record.rows, 1) / 2500.0),
        math.log(max(record.columns, 1) / 2500.0),
        math.log(max(record.columns, 1) / max(record.rows, 1)),
    ), dtype=torch.float32)


class StudyDataset(Dataset):
    def __init__(self, csv_path, image_root, image_size=320, local_canvas_size=640,
                 augment=False, max_studies=None, max_images_per_study=None,
                 progressive_augment=False, augmentation_ramp_start=3,
                 augmentation_ramp_end=15) -> None:
        self.records = read_studies(csv_path, image_root, max_studies=max_studies)
        self.image_size = image_size
        self.local_canvas_size = local_canvas_size
        self.augment = augment
        self.max_images_per_study = max_images_per_study
        self.progressive_augment = progressive_augment
        self.augmentation_ramp_start = augmentation_ramp_start
        self.augmentation_ramp_end = augmentation_ramp_end
        self.epoch = 1

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    @property
    def augmentation_strength(self) -> Optional[float]:
        if not self.augment or not self.progressive_augment:
            return None
        return progressive_augmentation_strength(
            self.epoch, self.augmentation_ramp_start, self.augmentation_ramp_end
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image_records = record.images[:self.max_images_per_study] if self.max_images_per_study else record.images
        global_images, local_patches = [], []
        for image_record in image_records:
            global_image, patches = load_multiscale_image(
                image_record.path, self.image_size, self.local_canvas_size, self.augment,
                self.augmentation_strength,
            )
            global_images.append(global_image)
            local_patches.append(patches)
        sample = {
            "subject_id": record.subject_id,
            "study_id": record.study_id,
            "global_images": global_images,
            "local_patches": local_patches,
            "view_ids": [r.view_id for r in image_records],
            "orientation_ids": [r.orientation_id for r in image_records],
            "numeric_metadata": [_numeric_metadata(r) for r in image_records],
        }
        if record.targets is not None:
            sample["targets"] = record.targets.clone()
        return sample


def collate_studies(samples: list[dict]) -> dict:
    global_images, local_patches, image_to_study = [], [], []
    view_ids, orientation_ids, numeric_metadata = [], [], []
    for study_index, sample in enumerate(samples):
        count = len(sample["global_images"])
        global_images.extend(sample["global_images"])
        local_patches.extend(sample["local_patches"])
        view_ids.extend(sample["view_ids"])
        orientation_ids.extend(sample["orientation_ids"])
        numeric_metadata.extend(sample["numeric_metadata"])
        image_to_study.extend([study_index] * count)
    batch = {
        "global_images": torch.stack(global_images),
        "local_patches": torch.stack(local_patches),
        "view_ids": torch.tensor(view_ids, dtype=torch.long),
        "orientation_ids": torch.tensor(orientation_ids, dtype=torch.long),
        "numeric_metadata": torch.stack(numeric_metadata),
        "image_to_study": torch.tensor(image_to_study, dtype=torch.long),
        "num_studies": len(samples),
        "subject_ids": [s["subject_id"] for s in samples],
        "study_ids": [s["study_id"] for s in samples],
    }
    if "targets" in samples[0]:
        batch["targets"] = torch.stack([s["targets"] for s in samples])
    return batch
