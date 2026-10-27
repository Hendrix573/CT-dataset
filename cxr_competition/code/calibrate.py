# -*- coding: utf-8 -*-
"""集成标定（只用验证集，测试集不参与）：
1) 读取各模型 valid_study_preds.csv，可选贪心挑选使 valid macro-AUC 最高的模型子集；
2) 对集成后概率逐类搜索 F1 最优阈值，写入 model_dir/thresholds.json；
推理时用分段线性单调映射把阈值对齐到 0.5 —— AUC 不变，默认 0.5 阈值下的 Macro-F1 提升。
注意：thresholds.json 须在合并 valid 训练之前生成（规则第3条）。"""
import os
import glob
import argparse

import numpy as np
import pandas as pd

from utils import NUM_CLASSES, CLASS_NAMES, macro_auc, macro_f1, best_thresholds, remap_probs, save_json

PCOLS = [f"p{k}" for k in range(1, NUM_CLASSES + 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="../model")
    ap.add_argument("--tta", type=int, default=1, help="推理是否开启水平翻转 TTA（写入 thresholds.json）")
    args = ap.parse_args()
    runs = sorted(glob.glob(os.path.join(args.model_dir, "*", "valid_study_preds.csv")))
    assert runs, "没有找到 valid_study_preds.csv，请先完成阶段1训练"
    preds, y = {}, None
    for r in runs:
        name = os.path.basename(os.path.dirname(r))
        df = pd.read_csv(r)
        preds[name] = df[PCOLS].values
        yy = np.load(os.path.join(os.path.dirname(r), "valid_study_labels.npy"))
        y = yy if y is None else y
        assert np.array_equal(y, yy), "各模型验证集顺序不一致"
        print(f"{name:40s} valid macro-AUC={macro_auc(y, preds[name])[0]:.4f}")

    ens = np.mean(list(preds.values()), 0)
    auc, aucs = macro_auc(y, ens)
    f1_05, _ = macro_f1(y, ens)
    ths = best_thresholds(y, ens)
    f1_opt, f1s = macro_f1(y, remap_probs(ens, ths))
    auc_after = macro_auc(y, remap_probs(ens, ths))[0]
    print(f"\n集成 {len(preds)} 模型: macro-AUC={auc:.4f} (重映射后 {auc_after:.4f})  "
          f"macro-F1@0.5: {f1_05:.4f} -> 重映射后 {f1_opt:.4f}")
    for k in range(NUM_CLASSES):
        print(f"  {k+1:2d} {CLASS_NAMES[k]:28s} AUC={aucs[k]:.4f} thr={ths[k]:.2f} F1={f1s[k]:.4f}")
    meta = dict(thresholds=ths, agg="meanmax", tta=bool(args.tta), models=list(preds.keys()),
                valid_macro_auc=auc, valid_macro_f1=f1_opt)
    save_json(meta, os.path.join(args.model_dir, "thresholds.json"))
    print("已写入", os.path.join(args.model_dir, "thresholds.json"))


if __name__ == "__main__":
    main()
