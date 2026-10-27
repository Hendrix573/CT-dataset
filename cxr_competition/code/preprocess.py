# -*- coding: utf-8 -*-
"""离线缓存：把原始大图（~3000px）确定性缩放为 cache_size 灰度 PNG，训练提速 10x+。
与推理同用 utils.load_gray，保证训练/推理像素一致。只处理 train / valid（测试集不缓存、不参与任何统计）。"""
import os
import argparse
from concurrent.futures import ProcessPoolExecutor

import cv2
import pandas as pd
from tqdm import tqdm

from utils import resolve_paths, load_gray


def _work(a):
    src, dst, size = a
    if os.path.exists(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    cv2.imwrite(dst, load_gray(src, size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="../../cxr_first")
    ap.add_argument("--cache_dir", default="../cache")
    ap.add_argument("--cache_size", type=int, default=576)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    for split in ["train", "valid"]:
        df = pd.read_csv(os.path.join(args.data_root, f"{split}.csv"))
        df = resolve_paths(df, os.path.join(args.data_root, split))
        out = os.path.join(args.cache_dir, str(args.cache_size), split)
        df["cache_path"] = [os.path.join(out, f"{d}.png") for d in df.Dicom_id]
        jobs = [(s, d, args.cache_size) for s, d in zip(df.path, df.cache_path)]
        if args.workers > 1:
            with ProcessPoolExecutor(args.workers) as ex:
                list(tqdm(ex.map(_work, jobs, chunksize=32), total=len(jobs), desc=split))
        else:
            for j in tqdm(jobs, desc=split):
                _work(j)
        df.to_csv(os.path.join(args.cache_dir, str(args.cache_size), f"{split}_index.csv"), index=False)
        print(split, len(df), "images,", df.Study_id.nunique(), "studies")


if __name__ == "__main__":
    main()
