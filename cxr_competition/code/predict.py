# -*- coding: utf-8 -*-
"""测试集推理 -> submission.csv（每个 Study 输出 10 行：类别 1..10 及概率）。
另写出：image_preds.csv（逐图原始概率，供 cost.py 一致性核验）
        test_meta.csv（逐图 ViewPosition/PatientOrientation，取自官方 test.csv，供 cost.py 使用同样的元数据）
每张图独立确定性预处理，batch=1，eval 模式，无任何测试集统计/自适应。"""
import os
import time
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import NUM_CLASSES, PCOLS, resolve_paths
from model import Predictor


def build_predictor(model_dir, fp16=True, device=None):
    """predict.py 与 cost.py 共用的模型加载入口"""
    return Predictor(model_dir, device=device, fp16=fp16)


def _meta(df, col):
    return df[col].values if col in df.columns else [None] * len(df)


def infer_images(predictor, df):
    vs, os_ = _meta(df, "ViewPosition"), _meta(df, "PatientOrientation")
    return np.stack([predictor.predict_image(p, v, o)
                     for p, v, o in tqdm(zip(df.path.values, vs, os_), total=len(df), desc="infer")])


def study_rows(predictor, df, img_probs, only_positive=False):
    rows = []
    df = df.reset_index(drop=True)
    for (sub, stu), g in df.groupby(["Subject_id", "Study_id"], sort=False):
        views = g.ViewPosition.values if "ViewPosition" in g.columns else None
        p = predictor.finalize_study(img_probs[g.index.values], views)
        for k in range(NUM_CLASSES):
            if only_positive and p[k] < 0.5:
                continue
            rows.append((sub, stu, k + 1, round(float(p[k]), 6)))
    return pd.DataFrame(rows, columns=["Subject_id", "Study_id", "Predict_class", "Probability"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="../../cxr_first")
    ap.add_argument("--split", default="test")
    ap.add_argument("--model_dir", default="../model")
    ap.add_argument("--out", default="../submission.csv")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--only_positive", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(args.data_root, f"{args.split}.csv"))
    df = resolve_paths(df, os.path.join(args.data_root, args.split))
    pred = build_predictor(args.model_dir, fp16=not args.fp32)
    print(f"models={len(pred.models)} precision={pred.precision} tta={pred.tta} agg={pred.agg} "
          f"frontal_w={pred.frontal_w} meta={pred.uses_meta} thresholds={'yes' if pred.thresholds else 'no'}")
    t0 = time.time()
    probs = infer_images(pred, df)
    print(f"{len(df)} images in {time.time()-t0:.1f}s ({(time.time()-t0)/len(df)*1000:.1f} ms/img)")

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    keep = [c for c in ["Dicom_id", "Subject_id", "Study_id", "ViewPosition", "PatientOrientation"] if c in df.columns]
    df[keep].to_csv(os.path.join(out_dir, "test_meta.csv"), index=False)
    img = df[["Dicom_id", "Subject_id", "Study_id"]].copy()
    img[PCOLS] = probs
    img.to_csv(os.path.join(out_dir, "image_preds.csv"), index=False, float_format="%.6f")
    sub = study_rows(pred, df, probs, args.only_positive)
    sub.to_csv(args.out, index=False, encoding="utf-8")
    print(f"写出 {args.out}: {sub.Study_id.nunique()} studies, {len(sub)} rows")


if __name__ == "__main__":
    main()
