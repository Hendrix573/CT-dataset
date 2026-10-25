from __future__ import annotations

import math

import torch


def binary_auc(target: torch.Tensor, score: torch.Tensor) -> float:
    target = target.to(torch.bool)
    positives = int(target.sum())
    negatives = int((~target).sum())
    if positives == 0 or negatives == 0:
        return math.nan
    order = torch.argsort(score)
    sorted_scores = score[order]
    ranks = torch.empty_like(score, dtype=torch.float64)
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum = ranks[target].sum().item()
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def binary_average_precision(target: torch.Tensor, score: torch.Tensor) -> float:
    target = target.to(torch.bool)
    positives = int(target.sum())
    if positives == 0:
        return math.nan
    order = torch.argsort(score, descending=True)
    sorted_target = target[order].to(torch.float64)
    precision = sorted_target.cumsum(0) / torch.arange(1, len(target) + 1)
    return float(precision[sorted_target.bool()].mean())


def multilabel_metrics(targets: torch.Tensor, probabilities: torch.Tensor) -> dict:
    aucs = [binary_auc(targets[:, i], probabilities[:, i]) for i in range(10)]
    aps = [binary_average_precision(targets[:, i], probabilities[:, i]) for i in range(10)]
    predictions = probabilities >= 0.5
    truth = targets.to(torch.bool)
    f1s = []
    for i in range(10):
        tp = int((predictions[:, i] & truth[:, i]).sum())
        fp = int((predictions[:, i] & ~truth[:, i]).sum())
        fn = int((~predictions[:, i] & truth[:, i]).sum())
        denominator = 2 * tp + fp + fn
        f1s.append((2 * tp / denominator) if denominator else math.nan)
    finite_auc = [value for value in aucs if math.isfinite(value)]
    finite_ap = [value for value in aps if math.isfinite(value)]
    finite_f1 = [value for value in f1s if math.isfinite(value)]
    return {
        "per_class_auc": aucs,
        "macro_auc": sum(finite_auc) / len(finite_auc) if finite_auc else math.nan,
        "auc_classes": len(finite_auc),
        "per_class_ap": aps,
        "macro_ap": sum(finite_ap) / len(finite_ap) if finite_ap else math.nan,
        "macro_f1_0.5": sum(finite_f1) / len(finite_f1) if finite_f1 else math.nan,
    }
