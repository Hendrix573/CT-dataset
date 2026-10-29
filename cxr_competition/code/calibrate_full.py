# -*- coding: utf-8 -*-
"""P4 标定：全量模型（train+valid 合并训练）没有留出预测，聚合方式和阈值改用同配方折实验的 OOF 预测来定。
  --oof_dirs   同分辨率、同配方的折实验目录（各含 oof_img_preds.csv）；只含官方训练集的折外预测
  --model_dirs 全量模型目录（各含 best.pth）
输出 out_dir：复制全量模型 + thresholds.json（thresholds / agg / frontal_w / tta）。"""
import os
import shutil
import argparse

import pandas as pd

from utils import PCOLS, NUM_CLASSES, CLASS_NAMES, study_labels, macro_f1, best_thresholds, remap_probs, save_json
from calibrate_cv import AGGS, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof_dirs", nargs="+", required=True)
    ap.add_argument("--model_dirs", nargs="+", required=True)
    ap.add_argument("--out_dir", default="../model_p4")
    ap.add_argument("--tta", type=int, default=1)
    args = ap.parse_args()

    oof = pd.concat([pd.read_csv(os.path.join(d, "oof_img_preds.csv")) for d in args.oof_dirs], ignore_index=True)
    assert oof.Dicom_id.is_unique, "OOF 图像重复：--oof_dirs 里不能有同一折的两个实验"
    oof = oof.sort_values("Dicom_id").reset_index(drop=True)
    a, mode, w = max((score(oof, m, w)[0], m, w) for m, w in AGGS)
    _, st = score(oof, mode, w)
    y = study_labels(oof)
    ths = best_thresholds(y, st[PCOLS].values)
    f1_raw = macro_f1(y, st[PCOLS].values)[0]
    f1_opt, f1s = macro_f1(y, remap_probs(st[PCOLS].values, ths))
    print(f"OOF 来源 {len(args.oof_dirs)} 个折实验，{len(st)} 个检查；聚合={mode} 正位权重={w:g} "
          f"OOF macro-AUC={a:.4f}  macro-F1@0.5 {f1_raw:.4f} -> 重映射后 {f1_opt:.4f}")
    for k in range(NUM_CLASSES):
        print(f"  {k+1:2d} {CLASS_NAMES[k]:28s} thr={ths[k]:.2f} OOF-F1={f1s[k]:.4f}")

    if os.path.isdir(args.out_dir):
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir)
    models = []
    for d in args.model_dirs:
        dst = os.path.join(args.out_dir, os.path.basename(os.path.normpath(d)))
        os.makedirs(dst)
        for f in ["best.pth", "cfg.json", "train_log.json"]:
            if os.path.exists(os.path.join(d, f)):
                shutil.copy(os.path.join(d, f), dst)
        models.append(os.path.basename(dst))
    save_json(dict(thresholds=ths, agg=mode, frontal_w=w, tta=bool(args.tta), models=models,
                   oof_macro_auc=a, oof_macro_f1=f1_opt,
                   oof_sources=[os.path.basename(os.path.normpath(d)) for d in args.oof_dirs],
                   note="全量模型用 train+valid 训练；轮数、聚合、阈值均在合并前由官方训练集 OOF 决定"),
              os.path.join(args.out_dir, "thresholds.json"))
    print(f"已写入 {args.out_dir}：{len(models)} 个模型 + thresholds.json (tta={bool(args.tta)})")


if __name__ == "__main__":
    main()
