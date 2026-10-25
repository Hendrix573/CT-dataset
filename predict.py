from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cxr_baseline import CLASS_NAMES, StudyDataset, StudyMultiInputNet, collate_studies
from train import move_batch


def parse_args():
    parser = argparse.ArgumentParser(description="Predict and export CXR submissions")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("cxr_first"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/predictions"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--max-studies", type=int)
    parser.add_argument("--max-images-per-study", type=int)
    return parser.parse_args()


def write_dense(path, ids, probabilities):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("Subject_id", "Study_id", "Predict_class", "Probability"))
        for (subject_id, study_id), scores in zip(ids, probabilities):
            for class_index, probability in enumerate(scores, start=1):
                writer.writerow((subject_id, study_id, class_index, f"{probability:.8f}"))


def write_thresholded(path, ids, probabilities, threshold):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("Subject_id", "Study_id", "Predict_class", "Probability"))
        for (subject_id, study_id), scores in zip(ids, probabilities):
            disease_positive = any(value >= threshold for value in scores[:9])
            for class_index, probability in enumerate(scores, start=1):
                if probability >= threshold and not (class_index == 10 and disease_positive):
                    writer.writerow((subject_id, study_id, class_index, f"{probability:.8f}"))


def main():
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if checkpoint.get("class_names") != list(CLASS_NAMES):
        raise ValueError("checkpoint class order mismatch")
    pre, arch = checkpoint["preprocessing"], checkpoint["architecture"]
    dataset = StudyDataset(
        args.data_root / "test.csv", args.data_root / "test",
        image_size=int(pre["image_size"]), local_canvas_size=int(pre["local_canvas_size"]),
        max_studies=args.max_studies, max_images_per_study=args.max_images_per_study,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, collate_fn=collate_studies,
                        pin_memory=device.type == "cuda")
    model = StudyMultiInputNet(arch["feature_dim"], arch["metadata_dim"],
                               arch["attention_heads"], arch["dropout"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    ids, probability_rows = [], []
    with torch.inference_mode():
        for batch in loader:
            model_batch, _ = move_batch(batch, device)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                logits = model(model_batch)["logits"]
            probability_rows.extend(logits.sigmoid().cpu().tolist())
            ids.extend(zip(batch["subject_ids"], batch["study_ids"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_dense(args.output_dir / "submission_dense.csv", ids, probability_rows)
    write_thresholded(args.output_dir / "submission_thresholded.csv", ids, probability_rows, args.threshold)
    print(f"wrote {len(ids)} Studies and {len(ids) * 10} dense probability rows")


if __name__ == "__main__":
    main()
