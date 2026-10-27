# -*- coding: utf-8 -*-
"""公共工具：标签解析、图像索引、确定性图像读取、指标、阈值重映射、检查级聚合。
训练 / 推理 / 计时脚本共用同一套函数，保证预处理完全一致。"""
import os
import glob
import json
import random

import cv2
import numpy as np
import pandas as pd
from PIL import Image

NUM_CLASSES = 10
CLASS_NAMES = [
    "Enlarged Cardiomediastinum", "Pneumothorax", "Consolidation", "Pneumonia",
    "Edema", "Cardiomegaly", "Atelectasis", "Lung Opacity", "Pleural Effusion",
    "No Finding",
]
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

cv2.setNumThreads(1)


def seed_everything(seed=42):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------- labels
def parse_labels(s):
    """'7;9' -> multi-hot(10)，类别 id 为 1..10"""
    y = np.zeros(NUM_CLASSES, dtype=np.float32)
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return y
    for t in str(s).replace(",", ";").split(";"):
        t = t.strip()
        if t:
            k = int(float(t))
            if 1 <= k <= NUM_CLASSES:
                y[k - 1] = 1.0
    return y


# ----------------------------------------------------------------- image index
def build_image_index(split_dir):
    """递归扫描目录 -> {文件名(无扩展名): 路径}。兼容 pic_xxx / image_xxx 命名。"""
    idx = {}
    for p in glob.glob(os.path.join(split_dir, "**", "*"), recursive=True):
        if p.lower().endswith(IMG_EXTS):
            idx[os.path.splitext(os.path.basename(p))[0]] = p
    return idx


def resolve_paths(df, split_dir):
    """为 csv 每行找到图像路径：优先 subject/study/Dicom_id.ext，其次全局索引。"""
    index = None
    paths = []
    for r in df.itertuples(index=False):
        found = None
        base = os.path.join(split_dir, str(r.Subject_id), str(r.Study_id), str(r.Dicom_id))
        for ext in IMG_EXTS:
            if os.path.exists(base + ext):
                found = base + ext
                break
        if found is None:
            if index is None:
                index = build_image_index(split_dir)
            found = index.get(str(r.Dicom_id))
        paths.append(found)
    df = df.copy()
    df["path"] = paths
    miss = df["path"].isna().sum()
    if miss:
        print(f"[WARN] {split_dir}: {miss} 张图像未找到，已丢弃")
        df = df[df["path"].notna()].reset_index(drop=True)
    return df


def scan_folder(folder):
    """cost.py 用：扫描与测试集结构一致的目录 -> DataFrame(Dicom_id, Subject_id, Study_id, path)"""
    rows = []
    for p in sorted(glob.glob(os.path.join(folder, "**", "*"), recursive=True)):
        if p.lower().endswith(IMG_EXTS):
            study_dir = os.path.dirname(p)
            rows.append(dict(Dicom_id=os.path.splitext(os.path.basename(p))[0],
                             Study_id=os.path.basename(study_dir),
                             Subject_id=os.path.basename(os.path.dirname(study_dir)),
                             path=p))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- image IO
def load_gray(path, cache_size):
    """确定性读取：JPEG 用 DCT 域降采样 (draft) 加速大图解码，再 INTER_AREA 缩放到 cache_size 方图。
    训练缓存与推理都调用本函数，保证像素级一致。"""
    with Image.open(path) as im:
        if im.format == "JPEG":
            im.draft("L", (cache_size, cache_size))  # 只会缩小到 >= 目标尺寸
        im = im.convert("L")
        arr = np.asarray(im)
    if arr.dtype != np.uint8:
        arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.resize(arr, (cache_size, cache_size), interpolation=cv2.INTER_AREA)


def to_model_input(gray, img_size):
    """uint8 方图 -> float32 (1,H,W)，单图自身 [0,1] 归一化 + 固定 ImageNet 灰度均值方差（规则允许）。"""
    if gray.shape[0] != img_size:
        gray = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
    x = gray.astype(np.float32) / 255.0
    x = (x - 0.449) / 0.226
    return x[None]


# ----------------------------------------------------------------- metrics
def _auc(y, s):
    """二分类 ROC-AUC（Mann-Whitney 秩统计，并列取平均秩），与 sklearn.roc_auc_score 一致，免去 sklearn 依赖。"""
    y = np.asarray(y).astype(bool)
    r = pd.Series(np.asarray(s, dtype=np.float64)).rank(method="average").values
    npos, nneg = y.sum(), (~y).sum()
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def _f1(y, pred):
    y, pred = np.asarray(y).astype(bool), np.asarray(pred).astype(bool)
    tp = (y & pred).sum()
    d = y.sum() + pred.sum()
    return float(2 * tp / d) if d > 0 else 0.0


def macro_auc(y, p):
    aucs = [(_auc(y[:, k], p[:, k]) if 0 < y[:, k].sum() < len(y) else np.nan) for k in range(y.shape[1])]
    return float(np.nanmean(aucs)), aucs


def macro_f1(y, p, thr=0.5):
    thr = np.broadcast_to(np.asarray(thr, dtype=np.float64), (y.shape[1],))
    f1s = [_f1(y[:, k], p[:, k] >= thr[k]) for k in range(y.shape[1])]
    return float(np.mean(f1s)), f1s


def best_thresholds(y, p):
    """逐类在验证集上搜索 F1 最优阈值。"""
    grid = np.linspace(0.02, 0.98, 97)
    return [float(grid[int(np.argmax([_f1(y[:, k], p[:, k] >= t) for t in grid]))]) for k in range(y.shape[1])]


def remap_probs(p, thr):
    """分段线性单调重映射：把类别最优阈值 t 映射到 0.5。
    单调变换不改变 AUC（主指标），但让主办方默认 0.5 阈值下的 F1（副指标）达到验证集最优。"""
    p = np.asarray(p, dtype=np.float64)
    t = np.clip(np.asarray(thr, dtype=np.float64), 1e-3, 1 - 1e-3)
    lo = 0.5 * p / t
    hi = 0.5 + 0.5 * (p - t) / (1 - t)
    return np.clip(np.where(p < t, lo, hi), 0.0, 1.0)


# ----------------------------------------------------------------- study aggregation
def aggregate_study(probs, mode="meanmax"):
    """同一次检查多张图 -> 检查级概率（确定性聚合，规则允许）。"""
    probs = np.asarray(probs, dtype=np.float64)
    if mode == "mean":
        return probs.mean(0)
    if mode == "max":
        return probs.max(0)
    return 0.5 * probs.mean(0) + 0.5 * probs.max(0)


def aggregate_df(img_df, prob_cols, mode="meanmax"):
    """img_df 含 Subject_id, Study_id 及概率列 -> 检查级 DataFrame（保持首次出现顺序）"""
    out = []
    for (sub, stu), g in img_df.groupby(["Subject_id", "Study_id"], sort=False):
        out.append([sub, stu, *aggregate_study(g[prob_cols].values, mode)])
    return pd.DataFrame(out, columns=["Subject_id", "Study_id", *prob_cols])


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
