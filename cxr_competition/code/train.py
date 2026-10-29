# -*- coding: utf-8 -*-
"""有监督微调（ImageNet 预训练骨干）。两种模式：
  P3 折模式：--folds_csv train_folds.csv --fold k  -> 在官方训练集的其余 4 折上训练，第 k 折选 checkpoint；
             训练结束后用 best 权重预测官方验证集（只做留出评估，不参与任何选择）。
  旧模式   ：不传 --folds_csv -> train 训练，valid 选 checkpoint（与 P0/P1 相同）。
P2：--use_meta 1 启用视角元数据分支；--meta_drop 训练时随机置为 MISSING，提升鲁棒性。
输出（out_dir）：best.pth, train_log.json, oof_img_preds.csv（逐图，选 checkpoint 用的那部分数据），
                holdout_img_preds.csv（折模式，官方验证集逐图），valid_study_preds.csv/labels.npy（兼容旧 calibrate.py）。"""
import os
import time
import math
import argparse

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils import (PCOLS, parse_labels, to_model_input, macro_auc, macro_f1, aggregate_df,
                   study_labels, seed_everything, save_json, meta_vector, MISSING_META)
from model import CXRNet, save_checkpoint, load_model


# ----------------------------------------------------------------- data
class CXRDataset(Dataset):
    def __init__(self, df, img_size, train, hflip=0.0, meta_drop=0.0):
        self.paths = df.cache_path.values
        lab = df.Predict_class.values if "Predict_class" in df.columns else [None] * len(df)
        self.y = np.stack([parse_labels(s) for s in lab])
        vp = df.ViewPosition.values if "ViewPosition" in df.columns else [None] * len(df)
        po = df.PatientOrientation.values if "PatientOrientation" in df.columns else [None] * len(df)
        self.meta = np.stack([meta_vector(v, o) for v, o in zip(vp, po)])
        self.img_size, self.train, self.hflip, self.meta_drop = img_size, train, hflip, meta_drop

    def __len__(self):
        return len(self.paths)

    def augment(self, g):
        s = g.shape[0]
        M = cv2.getRotationMatrix2D((s / 2, s / 2), np.random.uniform(-10, 10), np.random.uniform(0.85, 1.15))
        M[:, 2] += np.random.uniform(-0.06, 0.06, 2) * s
        g = cv2.warpAffine(g, M, (s, s), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        crop = int(s * np.random.uniform(0.85, 1.0))
        y0, x0 = np.random.randint(0, s - crop + 1, 2)
        g = cv2.resize(g[y0:y0 + crop, x0:x0 + crop], (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        g = g.astype(np.float32) * np.random.uniform(0.8, 1.2) + np.random.uniform(-20, 20)
        g = 255.0 * (np.clip(g, 0, 255) / 255.0) ** np.random.uniform(0.8, 1.25)
        if np.random.rand() < self.hflip:
            g = g[:, ::-1]
        if np.random.rand() < 0.3:
            for _ in range(np.random.randint(1, 4)):
                h, w = np.random.randint(self.img_size // 16, self.img_size // 6, 2)
                yy, xx = np.random.randint(0, self.img_size - h), np.random.randint(0, self.img_size - w)
                g[yy:yy + h, xx:xx + w] = 0
        return np.ascontiguousarray(g).clip(0, 255).astype(np.uint8)

    def __getitem__(self, i):
        g = cv2.imread(self.paths[i], cv2.IMREAD_GRAYSCALE)
        m = self.meta[i]
        if self.train:
            g = self.augment(g)
            if np.random.rand() < self.meta_drop:
                m = MISSING_META
        return torch.from_numpy(to_model_input(g, self.img_size)), torch.from_numpy(m), torch.from_numpy(self.y[i])


# ----------------------------------------------------------------- loss / ema
class AsymmetricLoss(nn.Module):
    """ASL (Ridnik et al. 2021)：多标签长尾常用，抑制大量易负样本。"""

    def __init__(self, gamma_neg=2.0, gamma_pos=0.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gn, self.gp, self.clip, self.eps = gamma_neg, gamma_pos, clip, eps

    def forward(self, logits, y):
        p = torch.sigmoid(logits.float())
        pn = (p - self.clip).clamp(min=0) if self.clip > 0 else p
        lp = y * torch.log(p.clamp(min=self.eps)) * (1 - p) ** self.gp
        ln = (1 - y) * torch.log((1 - pn).clamp(min=self.eps)) * pn ** self.gn
        return -(lp + ln).sum(1).mean()


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.is_floating_point():
                self.shadow[k].mul_(self.decay).add_(v.float(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model):
        model.load_state_dict({k: v.to(model.state_dict()[k].dtype) for k, v in self.shadow.items()})


@torch.no_grad()
def predict_loader(model, loader, device, amp, tta=True):
    model.eval()
    out = []
    for x, m, _ in loader:
        x, m = x.to(device, non_blocking=True), m.to(device, non_blocking=True)
        mm = m if model.use_meta else None
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            p = torch.sigmoid(model(x, mm).float())
            if tta:
                p = 0.5 * (p + torch.sigmoid(model(torch.flip(x, dims=[3]), mm).float()))
        out.append(p.cpu().numpy())
    return np.concatenate(out)


def img_pred_df(df, probs):
    cols = [c for c in ["Dicom_id", "Subject_id", "Study_id", "ViewPosition", "PatientOrientation", "Predict_class"]
            if c in df.columns]
    out = df[cols].copy().reset_index(drop=True)
    out[PCOLS] = probs
    return out


def evaluate(model, loader, df, device, amp, agg, frontal_w):
    img = img_pred_df(df, predict_loader(model, loader, device, amp))
    st = aggregate_df(img, PCOLS, agg, frontal_w)
    y = study_labels(img)
    auc, aucs = macro_auc(y, st[PCOLS].values)
    f1, _ = macro_f1(y, st[PCOLS].values)
    return auc, aucs, f1, st, y, img


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="../cache/576")
    ap.add_argument("--folds_csv", default=None, help="make_folds.py 输出；配合 --fold 使用")
    ap.add_argument("--fold", type=int, default=-1)
    ap.add_argument("--out_dir", default="../model/run")
    ap.add_argument("--backbone", default="tf_efficientnetv2_s.in21k_ft_in1k")
    ap.add_argument("--pretrained_path", default=None)
    ap.add_argument("--no_pretrained", action="store_true", help="仅冒烟测试使用")
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--warmup", type=float, default=1.0)
    ap.add_argument("--loss", default="asl", choices=["asl", "bce"])
    ap.add_argument("--hflip", type=float, default=0.5)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--drop_path", type=float, default=0.3)
    ap.add_argument("--drop_rate", type=float, default=0.2)
    ap.add_argument("--use_meta", type=int, default=1)
    ap.add_argument("--meta_drop", type=float, default=0.1)
    ap.add_argument("--agg", default="meanmax", choices=["mean", "max", "meanmax"])
    ap.add_argument("--frontal_w", type=float, default=1.0)
    ap.add_argument("--merge_valid", action="store_true", help="全量模式：train+valid 合并，固定轮数，无验证")
    ap.add_argument("--stop_epoch", type=int, default=0,
                    help="全量模式：学习率仍按 --epochs 的余弦计划，在第 stop_epoch 轮保存并停止（复现折实验最佳轮）")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_steps", type=int, default=0, help="冒烟测试：每 epoch 最多步数")
    args = ap.parse_args()

    seed_everything(args.seed + max(args.fold, 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    cache_size = int(os.path.basename(os.path.normpath(args.cache_dir)))
    holdout = None
    if args.folds_csv:
        folds = pd.read_csv(args.folds_csv)
        tr = folds[folds.fold != args.fold].reset_index(drop=True)
        va = folds[folds.fold == args.fold].reset_index(drop=True)
        holdout = pd.read_csv(os.path.join(args.cache_dir, "valid_index.csv"))
    else:
        tr = pd.read_csv(os.path.join(args.cache_dir, "train_index.csv"))
        va = pd.read_csv(os.path.join(args.cache_dir, "valid_index.csv"))
        if args.merge_valid:
            tr = pd.concat([tr, va], ignore_index=True)
    print(f"train imgs={len(tr)} select imgs={len(va)} holdout imgs={0 if holdout is None else len(holdout)} "
          f"fold={args.fold} use_meta={args.use_meta} device={device}", flush=True)

    pw = args.workers > 0
    dl_tr = DataLoader(CXRDataset(tr, args.img_size, True, args.hflip, args.meta_drop), batch_size=args.bs,
                       shuffle=True, num_workers=args.workers, pin_memory=amp, drop_last=True, persistent_workers=pw)
    dl_va = DataLoader(CXRDataset(va, args.img_size, False), batch_size=args.bs * 2, shuffle=False,
                       num_workers=args.workers, pin_memory=amp)

    use_meta = bool(args.use_meta)
    model = CXRNet(args.backbone, pretrained=not args.no_pretrained, pretrained_path=args.pretrained_path,
                   drop_rate=args.drop_rate, drop_path_rate=args.drop_path, use_meta=use_meta).to(device)
    ema = EMA(model, args.ema) if args.ema > 0 else None
    eval_model = CXRNet(args.backbone, pretrained=False, use_meta=use_meta).to(device) if ema else model
    crit = AsymmetricLoss() if args.loss == "asl" else nn.BCEWithLogitsLoss()
    head_ids = {id(p) for p in model.head.parameters()} | {id(model.p)}
    if use_meta:
        head_ids |= {id(p) for p in model.meta_fc.parameters()}
    head_params = [p for p in model.parameters() if id(p) in head_ids]
    opt = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": args.lr},
                             {"params": head_params, "lr": args.lr * 5}], weight_decay=args.wd)
    steps_ep = len(dl_tr) if not args.max_steps else min(len(dl_tr), args.max_steps)
    total, warm = steps_ep * args.epochs, int(steps_ep * args.warmup)
    base = [g["lr"] for g in opt.param_groups]
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    cfg = dict(backbone=args.backbone, img_size=args.img_size, cache_size=cache_size, epochs=args.epochs,
               agg=args.agg, frontal_w=args.frontal_w, merge_valid=args.merge_valid, seed=args.seed,
               fold=args.fold, lr=args.lr, wd=args.wd, drop_path=args.drop_path, drop_rate=args.drop_rate,
               loss=args.loss, hflip=args.hflip, use_meta=use_meta, meta_drop=args.meta_drop)
    os.makedirs(args.out_dir, exist_ok=True)
    save_json(cfg, os.path.join(args.out_dir, "cfg.json"))
    best, step, log = -1, 0, []
    for ep in range(args.epochs):
        model.train()
        t0, run = time.time(), 0.0
        for i, (x, m, y) in enumerate(dl_tr):
            if args.max_steps and i >= args.max_steps:
                break
            f = step / max(1, warm) if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
            for g, b in zip(opt.param_groups, base):
                g["lr"] = b * f
            x, m, y = x.to(device, non_blocking=True), m.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                loss = crit(model(x, m if use_meta else None), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            if ema:
                ema.update(model)
            run += loss.item()
            step += 1
        msg = dict(epoch=ep + 1, loss=run / max(1, steps_ep), time=round(time.time() - t0, 1))
        if ema:
            ema.copy_to(eval_model)
        if not args.merge_valid:
            auc, aucs, f1, st, y, img = evaluate(eval_model, dl_va, va, device, amp, args.agg, args.frontal_w)
            msg.update(val_auc=round(auc, 4), val_f1_05=round(f1, 4))
            if auc > best:
                best = auc
                save_checkpoint(eval_model, {**cfg, "best_epoch": ep + 1, "val_auc": auc}, args.out_dir)
                img.to_csv(os.path.join(args.out_dir, "oof_img_preds.csv"), index=False, float_format="%.6f")
                st.to_csv(os.path.join(args.out_dir, "valid_study_preds.csv"), index=False)
                np.save(os.path.join(args.out_dir, "valid_study_labels.npy"), y)
                msg["saved"] = True
                msg["per_class_auc"] = [round(a, 4) for a in aucs]
        elif ep + 1 == (args.stop_epoch or args.epochs):
            save_checkpoint(eval_model, {**cfg, "best_epoch": ep + 1, "stop_epoch": args.stop_epoch}, args.out_dir)
            msg["saved"] = True
        if not np.isfinite(msg["loss"]):
            raise SystemExit(f"loss 非有限值: {msg}")
        print(msg, flush=True)
        log.append(msg)
        save_json(log, os.path.join(args.out_dir, "train_log.json"))
        if args.merge_valid and args.stop_epoch and ep + 1 >= args.stop_epoch:
            break

    if holdout is not None:  # 用 best 权重预测官方验证集（仅留出评估）
        bm, _ = load_model(os.path.join(args.out_dir, "best.pth"), device)
        dl_ho = DataLoader(CXRDataset(holdout, args.img_size, False), batch_size=args.bs * 2, shuffle=False,
                           num_workers=args.workers, pin_memory=amp)
        img = img_pred_df(holdout, predict_loader(bm, dl_ho, device, amp))
        img.to_csv(os.path.join(args.out_dir, "holdout_img_preds.csv"), index=False, float_format="%.6f")
        st = aggregate_df(img, PCOLS, args.agg, args.frontal_w)
        print(f"holdout(官方验证集) 单折 macro-AUC = {macro_auc(study_labels(img), st[PCOLS].values)[0]:.4f}")
    print("done. best select macro-AUC =", round(best, 4) if best >= 0 else "n/a (merge_valid)")


if __name__ == "__main__":
    main()
