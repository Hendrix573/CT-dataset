# -*- coding: utf-8 -*-
"""P3 集成标定：全部选择只用官方训练集的 OOF（折外）预测，官方验证集只做留出评估。
  --group 名称=实验前缀   每组需有 <exp_dir>/<前缀>_f0..f{K-1}，各含 DONE / best.pth / oof_img_preds.csv / holdout_img_preds.csv
步骤：1) 拼接每组 OOF 逐图预测（每张训练图恰好一次）；
      2) 比较各组合（单组 / 多组平均）× 聚合方式（mean/max/meanmax × 正位权重 1/2/3）的 OOF 检查级 macro-AUC；
      3) 选最优（组合差距 <0.002 时优先成员多的）；在 OOF 上逐类求 F1 最优阈值；
      4) 打印官方验证集上的留出成绩（5 折模型平均，即真实部署形态）；
      5) 把所选组合的全部折模型复制到 --out_dir，写 thresholds.json。
  --only 名称…  强制只用指定组（耗时超限时降级用）"""
import os
import glob
import shutil
import argparse
import itertools

import numpy as np
import pandas as pd

from utils import (PCOLS, NUM_CLASSES, CLASS_NAMES, aggregate_df, study_labels, macro_auc, macro_f1,
                   best_thresholds, remap_probs, save_json)

AGGS = [(m, w) for m in ["meanmax", "mean", "max"] for w in [1.0, 2.0, 3.0] if not (m == "max" and w != 1.0)]


def load_group(exp_dir, prefix, nfolds):
    oof, ho, dirs = [], [], []
    for k in range(nfolds):
        d = os.path.join(exp_dir, f"{prefix}_f{k}")
        if not os.path.exists(os.path.join(d, "DONE")):
            raise SystemExit(f"缺少已完成的折: {d}")
        oof.append(pd.read_csv(os.path.join(d, "oof_img_preds.csv")))
        ho.append(pd.read_csv(os.path.join(d, "holdout_img_preds.csv")))
        dirs.append(d)
    oof = pd.concat(oof, ignore_index=True)
    assert oof.Dicom_id.is_unique, f"{prefix}: OOF 图像重复，请检查折划分"
    h = ho[0].copy()
    h[PCOLS] = np.mean([x[PCOLS].values for x in ho], 0)   # 5 折平均 = 部署形态
    return oof.sort_values("Dicom_id").reset_index(drop=True), h, dirs


def combine(frames):
    out = frames[0].copy()
    out[PCOLS] = np.mean([f[PCOLS].values for f in frames], 0)
    return out


def score(img, mode, w):
    st = aggregate_df(img, PCOLS, mode, w)
    return macro_auc(study_labels(img), st[PCOLS].values)[0], st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", default="../exp_p3")
    ap.add_argument("--group", action="append", required=True, help="名称=实验前缀，可多次")
    ap.add_argument("--nfolds", type=int, default=5)
    ap.add_argument("--out_dir", default="../model_p3")
    ap.add_argument("--tta", type=int, default=1)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    groups = dict(g.split("=", 1) for g in args.group)
    if args.only:
        groups = {k: v for k, v in groups.items() if k in args.only}
    data = {}
    for name, prefix in groups.items():
        oof, ho, dirs = load_group(args.exp_dir, prefix, args.nfolds)
        data[name] = (oof, ho, dirs)
        a, _ = score(oof, "meanmax", 1.0)
        b, _ = score(ho, "meanmax", 1.0)
        print(f"{name:8s} ({prefix})  OOF macro-AUC={a:.4f}   官方验证集(5折平均)={b:.4f}")
    ids = data[next(iter(data))][0].Dicom_id.values
    for n in data:
        assert np.array_equal(data[n][0].Dicom_id.values, ids), "各组 OOF 图像集合不一致"

    print("\n---- 组合 × 聚合方式（OOF 检查级 macro-AUC）----")
    names = list(data)
    results = []
    for r in range(1, len(names) + 1):
        for combo in itertools.combinations(names, r):
            img = combine([data[n][0] for n in combo])
            for mode, w in AGGS:
                a, _ = score(img, mode, w)
                results.append((a, combo, mode, w))
            best_c = max(x for x in results if x[1] == combo)
            print(f"  {'+'.join(combo):20s} 最佳 {best_c[0]:.4f}  ({best_c[2]}, 正位权重 {best_c[3]:g})")
    top = max(r[0] for r in results)
    cands = [r for r in results if r[0] >= top - 0.002]
    a, combo, mode, w = max(cands, key=lambda r: (len(r[1]), r[0]))
    print(f"\n==> 选择 {'+'.join(combo)}  聚合={mode} 正位权重={w:g}  OOF macro-AUC={a:.4f}（最高 {top:.4f}）")

    oof = combine([data[n][0] for n in combo])
    ho = combine([data[n][1] for n in combo])
    _, st = score(oof, mode, w)
    y = study_labels(oof)
    ths = best_thresholds(y, st[PCOLS].values)
    f1_raw = macro_f1(y, st[PCOLS].values)[0]
    f1_opt, _ = macro_f1(y, remap_probs(st[PCOLS].values, ths))
    ho_auc, ho_st = score(ho, mode, w)
    yh = study_labels(ho)
    _, ho_aucs = macro_auc(yh, ho_st[PCOLS].values)
    ho_f1_raw = macro_f1(yh, ho_st[PCOLS].values)[0]
    ho_f1, ho_f1s = macro_f1(yh, remap_probs(ho_st[PCOLS].values, ths))
    print(f"OOF:        macro-AUC={a:.4f}  macro-F1@0.5 {f1_raw:.4f} -> 重映射后 {f1_opt:.4f}")
    print(f"官方验证集: macro-AUC={ho_auc:.4f}  macro-F1@0.5 {ho_f1_raw:.4f} -> 重映射后 {ho_f1:.4f}   ← 最接近测试集的估计")
    for k in range(NUM_CLASSES):
        print(f"  {k+1:2d} {CLASS_NAMES[k]:28s} 验证集AUC={ho_aucs[k]:.4f} thr={ths[k]:.2f} 验证集F1={ho_f1s[k]:.4f}")

    if os.path.isdir(args.out_dir):
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir)
    models = []
    for n in combo:
        for d in data[n][2]:
            dst = os.path.join(args.out_dir, os.path.basename(d))
            os.makedirs(dst)
            shutil.copy(os.path.join(d, "best.pth"), dst)
            for f in ["cfg.json", "train_log.json"]:
                if os.path.exists(os.path.join(d, f)):
                    shutil.copy(os.path.join(d, f), dst)
            models.append(os.path.basename(d))
    save_json(dict(thresholds=ths, agg=mode, frontal_w=w, tta=bool(args.tta), models=models,
                   oof_macro_auc=a, oof_macro_f1=f1_opt, holdout_macro_auc=ho_auc, holdout_macro_f1=ho_f1,
                   note="阈值/聚合/组合均只由官方训练集 OOF 选出；官方验证集未参与训练与任何选择"),
              os.path.join(args.out_dir, "thresholds.json"))
    print(f"已写入 {args.out_dir}：{len(models)} 个模型 + thresholds.json")


if __name__ == "__main__":
    main()
