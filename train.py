from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cxr_baseline import CLASS_NAMES, StudyDataset, StudyMultiInputNet, collate_studies
from cxr_baseline.metrics import multilabel_metrics


CHINA_TIMEZONE = timezone(timedelta(hours=8), name="Asia/Shanghai")


def create_timestamped_run_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(CHINA_TIMEZONE).strftime("%Y%m%d_%H%M%S")
    candidate = root / timestamp
    suffix = 1
    while candidate.exists():
        candidate = root / f"{timestamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir()
    return candidate


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_run_config(run_dir, args, device, parameter_count, train_count, valid_count):
    config = {
        "created_at": datetime.now(CHINA_TIMEZONE).isoformat(),
        "run_dir": str(run_dir),
        "device": str(device),
        "parameters": parameter_count,
        "train_studies": train_count,
        "valid_studies": valid_count,
        "arguments": vars(args),
        "preprocessing": preprocessing_config(args),
        "architecture": architecture_config(args),
        "class_names": list(CLASS_NAMES),
    }
    (run_dir / "config.json").write_text(
        json.dumps(json_safe(config), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def append_metrics(run_dir: Path, report: dict) -> None:
    safe_report = json_safe(report)
    with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(safe_report, ensure_ascii=False, allow_nan=False) + "\n")

    csv_row = {
        "time": safe_report["time"], "epoch": safe_report["epoch"],
        "optimizer_step": safe_report["optimizer_step"], "learning_rate": safe_report["learning_rate"],
        "train_loss": safe_report["train_loss"],
        "train_eval_loss": safe_report["train_eval_loss"],
        "train_macro_auc": safe_report["train_macro_auc"],
        "train_auc_classes": safe_report["train_auc_classes"],
        "train_macro_ap": safe_report["train_macro_ap"],
        "train_macro_f1_0.5": safe_report["train_macro_f1_0.5"],
        "augmentation_strength": safe_report["augmentation_strength"],
        "valid_loss": safe_report["loss"],
        "macro_auc": safe_report["macro_auc"], "auc_classes": safe_report["auc_classes"],
        "macro_ap": safe_report["macro_ap"], "macro_f1_0.5": safe_report["macro_f1_0.5"],
        "best_macro_auc": safe_report["best_macro_auc"], "epoch_seconds": safe_report["epoch_seconds"],
    }
    for index, value in enumerate(safe_report["per_class_auc"], start=1):
        csv_row[f"auc_{index}"] = value
    for index, value in enumerate(safe_report["per_class_ap"], start=1):
        csv_row[f"ap_{index}"] = value
    for index, value in enumerate(safe_report["train_per_class_auc"] or [None] * 10, start=1):
        csv_row[f"train_auc_{index}"] = value
    for index, value in enumerate(safe_report["train_per_class_ap"] or [None] * 10, start=1):
        csv_row[f"train_ap_{index}"] = value
    csv_path = run_dir / "metrics.csv"
    write_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_row))
        if write_header:
            writer.writeheader()
        writer.writerow(csv_row)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the multi-input Study-level CXR model")
    parser.add_argument("--data-root", type=Path, default=Path("cxr_first"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/multiscale"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=320, help="Full-image branch size")
    parser.add_argument(
        "--local-canvas-size", type=int, default=640,
        help="Local canvas split into four 320x320 patches by default",
    )
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--metadata-dim", type=int, default=32)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument(
        "--local-token-grid", type=int, default=2,
        help="Spatial token grid retained inside each of the four local patches",
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=1, help="Studies per micro-batch")
    parser.add_argument("--accum-studies", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument(
        "--progressive-augment", action=argparse.BooleanOptionalAction, default=True,
        help="Ramp conservative CXR augmentation from mild to full strength",
    )
    parser.add_argument(
        "--dynamic-imaging-augment", action=argparse.BooleanOptionalAction, default=True,
        help="Randomly compose exposure and acquisition-style augmentations",
    )
    parser.add_argument("--augmentation-ramp-start", type=int, default=3)
    parser.add_argument("--augmentation-ramp-end", type=int, default=15)
    parser.add_argument(
        "--train-eval-interval", type=int, default=5,
        help="Evaluate the augmentation-free training set every N epochs; 0 disables it",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--amp", action="store_true", help="Use CUDA FP16")
    parser.add_argument("--max-train-studies", type=int)
    parser.add_argument("--max-valid-studies", type=int)
    parser.add_argument("--max-images-per-study", type=int)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args(argv)


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_optimizer(model, args):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (no_decay if parameter.ndim == 1 or name.endswith(".bias") else decay).append(parameter)
    return torch.optim.AdamW([
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=args.lr, betas=(0.9, 0.999), eps=1e-8)


def move_batch(batch, device):
    model_batch = {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
        if key not in {"targets", "subject_ids", "study_ids"}
    }
    targets = batch.get("targets")
    if targets is not None:
        targets = targets.to(device, non_blocking=True)
    return model_batch, targets


def divide_gradients(model, divisor):
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(divisor)


def optimizer_step(model, optimizer, scaler, accumulated_studies, amp_enabled):
    if amp_enabled:
        scaler.unscale_(optimizer)
    divide_gradients(model, accumulated_studies)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)


def train_epoch(model, loader, optimizer, scheduler, scaler, device, args):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    amp_enabled = args.amp and device.type == "cuda"
    total_loss = 0.0
    seen_studies = accumulated = 0
    for batch_index, batch in enumerate(loader):
        model_batch, targets = move_batch(batch, device)
        study_count = int(batch["num_studies"])
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(model_batch)["logits"]
            loss = F.binary_cross_entropy_with_logits(logits, targets)
        scaler.scale(loss * study_count).backward()
        total_loss += loss.item() * study_count
        seen_studies += study_count
        accumulated += study_count
        if accumulated >= args.accum_studies or batch_index + 1 == len(loader):
            optimizer_step(model, optimizer, scaler, accumulated, amp_enabled)
            scheduler.step()
            accumulated = 0
    return total_loss / max(1, seen_studies)


@torch.inference_mode()
def evaluate(model, loader, device, amp):
    model.eval()
    losses, total = 0.0, 0
    all_targets, all_probabilities = [], []
    for batch in loader:
        model_batch, targets = move_batch(batch, device)
        with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
            logits = model(model_batch)["logits"]
            loss = F.binary_cross_entropy_with_logits(logits, targets)
        count = int(batch["num_studies"])
        losses += loss.item() * count
        total += count
        all_targets.append(targets.cpu())
        all_probabilities.append(logits.sigmoid().cpu())
    metrics = multilabel_metrics(torch.cat(all_targets), torch.cat(all_probabilities))
    metrics["loss"] = losses / max(1, total)
    return metrics


def training_evaluation_report(metrics=None):
    """Prefix evaluation metrics so training and validation results stay unambiguous."""
    if metrics is None:
        return {
            "train_eval_loss": None,
            "train_per_class_auc": None,
            "train_macro_auc": None,
            "train_auc_classes": None,
            "train_per_class_ap": None,
            "train_macro_ap": None,
            "train_macro_f1_0.5": None,
        }
    return {
        "train_eval_loss": metrics["loss"],
        "train_per_class_auc": metrics["per_class_auc"],
        "train_macro_auc": metrics["macro_auc"],
        "train_auc_classes": metrics["auc_classes"],
        "train_per_class_ap": metrics["per_class_ap"],
        "train_macro_ap": metrics["macro_ap"],
        "train_macro_f1_0.5": metrics["macro_f1_0.5"],
    }


def preprocessing_config(args):
    return {
        "image_size": args.image_size,
        "local_canvas_size": args.local_canvas_size,
        "channels": 1,
        "normalization": "x=(x/127.5)-1",
        "quadrants": ["image-left-upper", "image-right-upper", "image-left-lower", "image-right-lower"],
        "training_augmentation": {
            "progressive": args.progressive_augment,
            "dynamic_imaging": args.dynamic_imaging_augment,
            "ramp_start_epoch": args.augmentation_ramp_start,
            "ramp_end_epoch": args.augmentation_ramp_end,
            "mild": {
                "rotation_degrees": 2.0, "translation_fraction": 0.01,
                "scale": [0.99, 1.01], "brightness": [0.98, 1.02],
                "contrast": [0.98, 1.02], "gamma": [1.0, 1.0],
            },
            "full": {
                "rotation_degrees": 5.0 if args.dynamic_imaging_augment else 7.0,
                "translation_fraction": 0.03 if args.dynamic_imaging_augment else 0.04,
                "scale": ([0.97, 1.03] if args.dynamic_imaging_augment else [0.95, 1.05]),
                "brightness": [0.90, 1.10],
                "contrast": [0.85, 1.15], "gamma": [0.90, 1.10],
            },
            "dynamic_policy": {
                "clean_probability": [0.30, 0.15],
                "affine_probability": [0.50, 0.70],
                "exposure_operations_per_image": [1, 2],
                "resolution_degradation_probability": [0.0, 0.20],
                "resolution_scale_at_full_strength": [0.65, 0.95],
                "gaussian_noise_probability": [0.0, 0.25],
                "gaussian_noise_std": [0.005, 0.025],
                "blur_probability": [0.0, 0.10],
                "sharpen_probability": [0.0, 0.10],
            },
        },
    }


def architecture_config(args):
    return {
        "name": "StudyMultiInputNet",
        "feature_dim": args.feature_dim,
        "metadata_dim": args.metadata_dim,
        "attention_heads": args.attention_heads,
        "dropout": args.dropout,
        "local_token_grid": args.local_token_grid,
        "local_tokens_per_patch": args.local_token_grid ** 2,
        "local_tokens_per_image": 4 * args.local_token_grid ** 2,
        "metadata_inputs": ["ViewPosition", "PatientOrientation", "Rows", "Columns", "aspect_ratio"],
    }


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_auc, args):
    payload = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
        "epoch": epoch, "optimizer_step": scheduler.last_epoch,
        "best_macro_auc": best_auc, "class_names": list(CLASS_NAMES),
        "preprocessing": preprocessing_config(args), "architecture": architecture_config(args),
        "config": vars(args) | {"data_root": str(args.data_root), "output_dir": str(args.output_dir),
                                  "resume": str(args.resume) if args.resume else None},
        "torch_rng_state": torch.get_rng_state(), "python_rng_state": random.getstate(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    torch.save(payload, path)


def restore_checkpoint(path, model, optimizer, scheduler, scaler, args):
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("class_names") != list(CLASS_NAMES):
        raise ValueError("checkpoint class order mismatch")
    if checkpoint.get("preprocessing") != preprocessing_config(args):
        raise ValueError("checkpoint preprocessing differs from current arguments")
    if checkpoint.get("architecture") != architecture_config(args):
        raise ValueError("checkpoint architecture differs from current arguments")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    random.setstate(checkpoint["python_rng_state"])
    if torch.cuda.is_available() and "cuda_rng_state_all" in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
    return int(checkpoint["epoch"]) + 1, float(checkpoint["best_macro_auc"])


def main(argv=None):
    args = parse_args(argv)
    if args.local_canvas_size % 2:
        raise ValueError("--local-canvas-size must be even")
    if args.local_token_grid < 1:
        raise ValueError("--local-token-grid must be positive")
    if args.train_eval_interval < 0:
        raise ValueError("--train-eval-interval must be non-negative")
    if args.progressive_augment:
        if args.augmentation_ramp_start < 0:
            raise ValueError("--augmentation-ramp-start must be non-negative")
        if args.augmentation_ramp_end <= args.augmentation_ramp_start:
            raise ValueError("--augmentation-ramp-end must be greater than ramp start")
    seed_everything(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    if args.resume:
        if not args.resume.is_file():
            raise FileNotFoundError(f"resume checkpoint does not exist: {args.resume}")
        run_dir = args.resume.resolve().parent
    else:
        run_dir = create_timestamped_run_dir(args.output_dir)
    args.output_dir = run_dir
    common = dict(image_size=args.image_size, local_canvas_size=args.local_canvas_size,
                  max_images_per_study=args.max_images_per_study)
    train_data = StudyDataset(args.data_root / "train.csv", args.data_root / "train",
                              augment=True, max_studies=args.max_train_studies,
                              progressive_augment=args.progressive_augment,
                              augmentation_ramp_start=args.augmentation_ramp_start,
                              augmentation_ramp_end=args.augmentation_ramp_end,
                              dynamic_imaging_augment=args.dynamic_imaging_augment,
                              **common)
    train_eval_data = None
    if args.train_eval_interval:
        # This creates only a compact CSV index; image pixels are still loaded on demand.
        train_eval_data = StudyDataset(
            args.data_root / "train.csv", args.data_root / "train",
            augment=False, max_studies=args.max_train_studies, **common,
        )
    valid_data = StudyDataset(args.data_root / "valid.csv", args.data_root / "valid",
                              augment=False, max_studies=args.max_valid_studies, **common)
    loader_args = dict(batch_size=args.batch_size, num_workers=args.workers,
                       collate_fn=collate_studies, pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_data, shuffle=True, **loader_args)
    train_eval_loader = (
        DataLoader(train_eval_data, shuffle=False, **loader_args)
        if train_eval_data is not None else None
    )
    valid_loader = DataLoader(valid_data, shuffle=False, **loader_args)
    model = StudyMultiInputNet(args.feature_dim, args.metadata_dim,
                               args.attention_heads, args.dropout,
                               args.local_token_grid).to(device)
    optimizer = make_optimizer(model, args)
    updates_per_epoch = max(1, math.ceil(len(train_data) / args.accum_studies))
    total_updates = max(1, updates_per_epoch * args.epochs)
    warmup_updates = round(updates_per_epoch * args.warmup_epochs)

    def lr_factor(step):
        if warmup_updates and step < warmup_updates:
            return max(1e-6, (step + 1) / warmup_updates)
        progress = (step - warmup_updates) / max(1, total_updates - warmup_updates)
        return max(1e-6 / args.lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    start_epoch, best_auc = 1, -math.inf
    if args.resume:
        start_epoch, best_auc = restore_checkpoint(
            args.resume, model, optimizer, scheduler, scaler, args
        )
    parameter_count = sum(p.numel() for p in model.parameters())
    write_run_config(run_dir, args, device, parameter_count, len(train_data), len(valid_data))
    print(json.dumps({"run_dir": str(run_dir), "device": str(device), "parameters": parameter_count,
                      "train_studies": len(train_data), "valid_studies": len(valid_data),
                      "local_tokens_per_image": 4 * args.local_token_grid ** 2,
                      "progressive_augment": args.progressive_augment,
                      "dynamic_imaging_augment": args.dynamic_imaging_augment,
                      "augmentation_ramp": [args.augmentation_ramp_start,
                                             args.augmentation_ramp_end]}))
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_started = time.perf_counter()
        train_data.set_epoch(epoch)
        augmentation_strength = train_data.augmentation_strength
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, args
        )
        train_eval_metrics = None
        if train_eval_loader is not None and epoch % args.train_eval_interval == 0:
            print(json.dumps({"epoch": epoch, "phase": "train_eval"}))
            train_eval_metrics = evaluate(model, train_eval_loader, device, args.amp)
        metrics = evaluate(model, valid_loader, device, args.amp)
        if math.isfinite(metrics["macro_auc"]) and metrics["macro_auc"] > best_auc:
            best_auc = metrics["macro_auc"]
            save_checkpoint(run_dir / "best.pt", model, optimizer, scheduler, scaler, epoch, best_auc, args)
        report = {
            "time": datetime.now(CHINA_TIMEZONE).isoformat(),
            "epoch": epoch,
            "optimizer_step": scheduler.last_epoch,
            "learning_rate": scheduler.get_last_lr()[0],
            "train_loss": train_loss,
            "augmentation_strength": augmentation_strength,
            **training_evaluation_report(train_eval_metrics),
            **metrics,
            "best_macro_auc": best_auc,
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        append_metrics(run_dir, report)
        print(json.dumps(json_safe(report), ensure_ascii=False, allow_nan=False))
        save_checkpoint(run_dir / "last.pt", model, optimizer, scheduler, scaler, epoch, best_auc, args)
    return run_dir / "last.pt"


if __name__ == "__main__":
    main()
