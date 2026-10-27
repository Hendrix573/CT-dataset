# -*- coding: utf-8 -*-
"""推理耗时 + 结果一致性核验（规则第7条 / 补充说明第九条）。
用法: python cost.py submission.csv pictures_100 pictures_5 [--model_dir model]
- 与 predict.py 导入同一个 build_predictor / Predictor.predict_image / finalize_study；
- batch=1 串行，先用 pictures_5 预热（不计时），再对 pictures_100 连续跑 3 轮；
- 单张计时 = 读图 -> 预处理 -> 前向 -> 后处理 -> 写出该图预测结果；
- 一致性：逐张与 submission.csv 同目录下的 image_preds.csv 比对（|Δp|<=1e-3、0.5 二值判定、类别概率排序），
  并对 100 张中完整覆盖的检查，再与 submission.csv 检查级结果比对。"""
import os
import sys
import time
import argparse

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "code"))
from utils import NUM_CLASSES, scan_folder  # noqa: E402
from predict import build_predictor  # noqa: E402

TOL = 1e-3
PCOLS = [f"p{k}" for k in range(1, NUM_CLASSES + 1)]


def same(a, b):
    """样本级一致：概率误差<=1e-3、0.5 二值判定一致、排序一致（差值<=1e-3 的并列不计）"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.max(np.abs(a - b)) > TOL:
        return False
    if not np.array_equal(a >= 0.5, b >= 0.5):
        return False
    for i in range(len(a)):
        for j in range(len(a)):
            if (a[i] - a[j]) > TOL and (b[i] - b[j]) < -TOL:
                return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("pictures_100")
    ap.add_argument("pictures_5")
    ap.add_argument("--model_dir", default=os.path.join(HERE, "model"))
    ap.add_argument("--image_preds", default=None)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--fp32", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    pred = build_predictor(args.model_dir, fp16=not args.fp32)  # 模型加载不计时
    sync = (lambda: torch.cuda.synchronize()) if pred.device.type == "cuda" else (lambda: None)
    warm = scan_folder(args.pictures_5)
    test = scan_folder(args.pictures_100)
    print(f"[INFO] 模型 {len(pred.models)} 个, 精度 {pred.precision}, TTA={pred.tta}, 设备 {pred.device}; "
          f"预热 {len(warm)} 张, 计时 {len(test)} 张")
    for p in warm.path:
        pred.predict_image(p)
    sync()

    out_txt = os.path.join(HERE, "cost_outputs.csv")
    times = np.zeros((args.rounds, len(test)))
    probs = np.zeros((len(test), NUM_CLASSES))
    with open(out_txt, "w", encoding="utf-8") as fo:
        for r in range(args.rounds):
            for i, row in enumerate(test.itertuples(index=False)):
                t0 = time.perf_counter()
                p = pred.predict_image(row.path)          # 读图+预处理+前向
                q = pred.finalize(p)                      # 后处理（阈值重映射）
                fo.write(f"{r},{row.Dicom_id}," + ",".join(f"{v:.6f}" for v in q) + "\n")
                fo.flush()                                # 写出最终预测结果
                sync()
                times[r, i] = time.perf_counter() - t0
                probs[i] = p
    per_img = times.mean(0) * 1000
    avg = float(per_img.mean())

    # ------------------------------------------------ consistency
    sub = pd.read_csv(args.submission)
    ip_path = args.image_preds or os.path.join(os.path.dirname(os.path.abspath(args.submission)), "image_preds.csv")
    bad, checked_img, bad_study, checked_study = set(), 0, 0, 0
    if os.path.exists(ip_path):
        ip = pd.read_csv(ip_path).set_index("Dicom_id")
        for i, row in enumerate(test.itertuples(index=False)):
            if row.Dicom_id not in ip.index:
                bad.add(row.Dicom_id)
                continue
            checked_img += 1
            ref = ip.loc[row.Dicom_id, PCOLS].values.astype(float)
            if not same(pred.finalize(probs[i]), pred.finalize(ref)):
                bad.add(row.Dicom_id)
        counts = ip.groupby("Study_id").size()
    else:
        print(f"[WARN] 未找到 {ip_path}，仅做检查级比对")
        counts = None
    test = test.assign(idx=range(len(test)))
    for stu, g in test.groupby("Study_id", sort=False):
        if counts is not None and counts.get(stu, -1) != len(g):
            continue  # 该检查的图未全部在 100 张中，检查级结果无法单独复现，已由逐图比对覆盖
        ref = sub[sub.Study_id == stu].sort_values("Predict_class")
        if len(ref) == 0:
            bad.update(g.Dicom_id)
            continue
        checked_study += 1
        mine = pred.finalize_study(probs[g.idx.values])
        refv = np.full(NUM_CLASSES, np.nan)
        refv[ref.Predict_class.values - 1] = ref.Probability.values
        m = ~np.isnan(refv)
        if not same(mine[m], refv[m]):
            bad_study += 1
            bad.update(g.Dicom_id)

    print(f"[DETAIL] 逐图比对 {checked_img} 张；检查级比对 {checked_study} 个检查，不一致 {bad_study} 个；"
          f"单张耗时 min/median/max = {per_img.min():.1f}/{np.median(per_img):.1f}/{per_img.max():.1f} ms；"
          f"逐图输出见 {out_txt}")
    status = "未超过上限" if avg <= 100 else ("明显超限(100~150ms)" if avg <= 150 else "严重超限(>150ms)")
    print(f"[CONCLUSION] 结论：{len(test)} 张样本、batch=1 串行、连续{args.rounds}轮；单张端到端墙钟时间先按每张"
          f"{args.rounds}轮取均值，再对{len(test)}张取平均，最终平均耗时 {avg:.0f} ms，{status}；"
          f"预测结果与 submission.csv 比对 {len(test)} 张，不一致 {len(bad)} 张；推理精度 {pred.precision}。")


if __name__ == "__main__":
    main()
