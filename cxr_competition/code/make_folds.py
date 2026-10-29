# -*- coding: utf-8 -*-
"""P3：按患者(Subject_id)分组 + 多标签分层的 K 折划分，只划分官方训练集。
官方验证集保持独立，作为留出集评估最终集成（不并入训练，不参与任何选择，天然满足规则第3条）。
输出: <cache_dir>/train_folds.csv（train_index.csv + fold 列）"""
import os
import argparse

import numpy as np
import pandas as pd

from utils import parse_labels, NUM_CLASSES, CLASS_NAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="../cache/576")
    ap.add_argument("--nfolds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()
    df = pd.read_csv(os.path.join(args.cache_dir, "train_index.csv"))
    st = df.groupby(["Subject_id", "Study_id"], sort=False).Predict_class.first().reset_index()
    Y = np.stack([parse_labels(s) for s in st.Predict_class])
    st_y = pd.DataFrame(Y, columns=range(NUM_CLASSES))
    st_y["Subject_id"] = st.Subject_id.values
    subj = st_y.groupby("Subject_id").sum()            # 每个患者各类阳性检查数
    n_st = st.groupby("Subject_id").size().reindex(subj.index).values
    S = subj.values
    tot = S.sum(0) + 1e-9
    rng = np.random.default_rng(args.seed)
    rarity = (S / tot).max(1) + rng.random(len(S)) * 1e-6   # 先放含稀有类的患者
    order = np.argsort(-rarity)
    fc = np.zeros((args.nfolds, NUM_CLASSES))
    fn = np.zeros(args.nfolds)
    target_n = n_st.sum() / args.nfolds
    fold_of = np.zeros(len(S), dtype=int)
    for i in order:
        pos = S[i] > 0
        if pos.any():
            score = (fc[:, pos] / tot[pos]).max(1) + 0.5 * fn / target_n
        else:
            score = fn / target_n
        k = int(np.argmin(score + rng.random(args.nfolds) * 1e-9))
        fold_of[i] = k
        fc[k] += S[i]
        fn[k] += n_st[i]
    m = dict(zip(subj.index, fold_of))
    df["fold"] = df.Subject_id.map(m).astype(int)
    out = os.path.join(args.cache_dir, "train_folds.csv")
    df.to_csv(out, index=False)
    print(f"写出 {out}")
    print("fold  studies  images  " + "  ".join(f"c{k+1}" for k in range(NUM_CLASSES)))
    for k in range(args.nfolds):
        print(f"{k:4d}  {int(fn[k]):7d}  {(df.fold == k).sum():6d}  " + "  ".join(f"{int(v):3d}" for v in fc[k]))
    assert df.groupby("Subject_id").fold.nunique().max() == 1, "同一患者跨折"
    print("检查：无患者跨折。类别:", ", ".join(f"c{k+1}={n}" for k, n in enumerate(CLASS_NAMES)))


if __name__ == "__main__":
    main()
