"""Streaming metadata audit for cxr_first (does not decode the image corpus)."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def numeric_summary(total, count, minimum, maximum):
    return {
        "min": minimum if count else None,
        "mean": total / count if count else None,
        "max": maximum if count else None,
    }


def scan_split(root: Path, split: str):
    csv_path = root / f"{split}.csv"
    image_root = root / split
    subjects = set()
    studies = {}
    dicom_ids = set()
    duplicate_images = 0
    inconsistent_labels = []
    missing_images = []
    missing_image_count = 0
    views, orientations = Counter(), Counter()
    row_total = column_total = 0
    ratio_total = 0.0
    row_min = column_min = None
    row_max = column_max = None
    ratio_min = ratio_max = None
    image_count = 0

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        has_labels = "Predict_class" in (reader.fieldnames or ())
        for row in reader:
            image_count += 1
            subject_id, study_id = row["Subject_id"], row["Study_id"]
            key = (subject_id, study_id)
            subjects.add(subject_id)
            label = row.get("Predict_class") if has_labels else None
            if key not in studies:
                studies[key] = {"images": 0, "label": label}
            elif studies[key]["label"] != label:
                inconsistent_labels.append(list(key))
            studies[key]["images"] += 1

            dicom_key = (subject_id, study_id, row["Dicom_id"])
            duplicate_images += dicom_key in dicom_ids
            dicom_ids.add(dicom_key)
            views[row.get("ViewPosition") or "<missing>"] += 1
            orientations[row.get("PatientOrientation") or "<missing>"] += 1
            height, width = int(row["Rows"]), int(row["Columns"])
            row_total += height
            column_total += width
            ratio = width / height
            ratio_total += ratio
            row_min = height if row_min is None else min(row_min, height)
            row_max = height if row_max is None else max(row_max, height)
            column_min = width if column_min is None else min(column_min, width)
            column_max = width if column_max is None else max(column_max, width)
            ratio_min = ratio if ratio_min is None else min(ratio_min, ratio)
            ratio_max = ratio if ratio_max is None else max(ratio_max, ratio)

            jpg = image_root / subject_id / study_id / f"{row['Dicom_id']}.jpg"
            if not jpg.is_file() and not jpg.with_suffix(".png").is_file():
                missing_image_count += 1
                if len(missing_images) < 100:
                    missing_images.append(str(jpg))

    images_per_study = Counter(record["images"] for record in studies.values())
    result = {
        "patients": len(subjects),
        "studies": len(studies),
        "images": image_count,
        "images_per_study": {str(key): value for key, value in sorted(images_per_study.items())},
        "height": numeric_summary(row_total, image_count, row_min, row_max),
        "width": numeric_summary(column_total, image_count, column_min, column_max),
        "width_to_height_ratio": numeric_summary(
            ratio_total, image_count, ratio_min, ratio_max
        ),
        "view_position": dict(sorted(views.items())),
        "view_position_missing_rate": views["<missing>"] / image_count,
        "patient_orientation": dict(sorted(orientations.items())),
        "patient_orientation_missing_rate": orientations["<missing>"] / image_count,
        "missing_image_count": missing_image_count,
        "missing_image_examples": missing_images,
        "duplicate_image_rows": duplicate_images,
        "inconsistent_study_label_count": len(set(map(tuple, inconsistent_labels))),
        "support_devices": "column unavailable in this dataset",
    }
    if has_labels:
        positives = [0] * 10
        no_finding_conflicts = []
        for key, record in studies.items():
            class_ids = {int(value) for value in record["label"].split(";")}
            for class_id in class_ids:
                positives[class_id - 1] += 1
            if 10 in class_ids and any(value < 10 for value in class_ids):
                no_finding_conflicts.append(list(key))
        result["labels"] = {
            str(index + 1): {
                "positive_studies": value,
                "negative_studies": len(studies) - value,
                "invalid_studies": 0,
            }
            for index, value in enumerate(positives)
        }
        result["no_finding_conflict_count"] = len(no_finding_conflicts)
    return result, subjects


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("cxr_first"))
    parser.add_argument("--output", type=Path, default=Path("data_audit.json"))
    args = parser.parse_args()
    report = {"note": "CSV metadata was streamed; image pixels were not decoded."}
    subject_sets = {}
    for split in ("train", "valid", "test"):
        report[split], subject_sets[split] = scan_split(args.data_root, split)
    report["subject_intersections"] = {
        "train_valid": len(subject_sets["train"] & subject_sets["valid"]),
        "train_test": len(subject_sets["train"] & subject_sets["test"]),
        "valid_test": len(subject_sets["valid"] & subject_sets["test"]),
    }
    report["image_integrity"] = {
        "corrupt_files": "not checked; run a separate full image decode audit before final training",
        "duplicate_file_content": "not checked; hashing the full image corpus was intentionally avoided",
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
