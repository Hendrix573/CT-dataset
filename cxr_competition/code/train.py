# -*- coding: utf-8 -*-
"""有监督微调（ImageNet 预训练骨干）。
阶段1：train 训练、valid 选 checkpoint，保存 valid 检查级预测供阈值/集成标定；
阶段2（可选）：--merge_valid，按阶段1确定的 epoch 数在 train+valid 上重训，不做任何基于 valid 的选择（规则第3条）。"""
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

from utils import (NUM_CLASSES, parse_labels, to_model_input, macro_auc, macro_f1,
                   aggregate_df, seed_everything, save_json)
from model import CXRNet, save_checkpoint

PCOLS = [f"p{k}" for k in range(1, NUM_CLASSES + 1)]


# ----------------------------------------------------------------- data
class CXRDataset(Dataset):
    def __init__(self, df, img_size, train, hflip=0.0):
        self.paths = df.cache_path.values
        self.y = np.stack([parse_labels(s) for s in df.Predict_class.values])
        self.img_size, self.train, self.hflip = img_size, train, hflip

    def __len__(self):
        return len(self.paths)

    def augment(self, g):
        s = g.shape[0]
        ang = np.random.uniform(-10, 10)
        sc = np.random.uniform(0.85, 1.15)
        M = cv2.getRotationMatrix2D((s / 2, s / 2), ang, sc)
        M[:, 2] += np.random.uniform(-0.06, 0.06, 2) * s
        g = cv2.warpAffine(g, M, (s, s), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # 随机裁剪到 img_size（等价 RandomResizedCrop 的温和版本）
        crop = int(s * np.random.uniform(0.85, 1.0))
        y0, x0 = np.random.randint(0, s - crop + 1, 2)
        g = cv2.resize(g[y0:y0 + crop, x0:x0 + crop], (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        g = g.astype(np.float32)
        g = g * np.random.uniform(0.8, 1.2) + np.random.uniform(-20, 20)  # 对比度/亮度
        g = np.clip(g, 0, 255)
        gamma = np.random.uniform(0.8, 1.25)
        g = 255.0 * (g / 255.0) ** gamma
        if np.random.rand() < self.hflip:
            g = g[:, ::-1]
        if np.random.rand() < 0.3:  # CoarseDropout
            for _ in range(np.random.randint(1, 4)):
                h, w = np.random.randint(self.img_size // 16, self.img_size // 6, 2)
                yy, xx = np.random.randint(0, self.img_size - h), np.random.randint(0, self.img_size - w)
                g[yy:yy + h, xx:xx + w] = 0
        return np.ascontiguousarray(g).clip(0, 255).astype(np.uint8)

    def __getitem__(self, i):
        g = cv2.imread(self.paths[i], cv2.IMREAD_GRAYSCALE)
        if self.train:
            g = self.augment(g)
        return torch.from_numpy(to_model_input(g, self.img_size)), torch.from_numpy(self.y[i])


# ----------------------------------------------------------------- loss
class AsymmetricLoss(nn.Module):
    """ASL (Ridnik et al. 2021)：多标签长尾常用，抑制大量易负样本。gamma_neg=0,clip=0 时退化为 BCE。"""

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
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            p = torch.sigmoid(model(x).float())
            if tta:
                p = 0.5 * (p + torch.sigmoid(model(torch.flip(x, dims=[3])).float()))
        out.append(p.cpu().numpy())
    return np.concatenate(out)


def evaluate(model, loader, df, device, amp, agg):
    p = predict_loader(model, loader, device, amp)
    img = df[["Subject_id", "Study_id"]].copy()
    img[PCOLS] = p
    st = aggregate_df(img, PCOLS, agg)
    ylab = df.groupby(["Subject_id", "Study_id"], sort=False).Predict_class.first()
    y = np.stack([parse_labels(s) for s in ylab.values])
    auc, aucs = macro_auc(y, st[PCOLS].values)
    f1, _ = macro_f1(y, st[PCOLS].values)
    return auc, aucs, f1, st, y


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="../cache/576")
    ap.add_argument("--out_dir", default="../model/convnext_small_512")
    ap.add_argument("--backbone", default="convnext_small.fb_in22k_ft_in1k")
    ap.add_argument("--pretrained_path", default=None, help="离线预训练权重文件（safetensors/pth），不填则从 HF Hub 下载")
    ap.add_argument("--no_pretrained", action="store_true", help="仅冒烟测试使用")
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=float, default=1.0, help="warmup epoch 数")
    ap.add_argument("--loss", default="asl", choices=["asl", "bce"])
    ap.add_argument("--hflip", type=float, default=0.5)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--agg", default="meanmax", choices=["mean", "max", "meanmax"])
    ap.add_argument("--merge_valid", action="store_true", help="阶段2：train+valid 训练固定 epochs，保存最后一轮")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_steps", type=int, default=0, help="冒烟测试：每 epoch 最多步数")
    args = ap.parse_args()

    if args.no_pretrained and args.pretrained_path:
        ap.error("--no_pretrained 与 --pretrained_path 不能同时使用")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    cache_size = int(os.path.basename(os.path.normpath(args.cache_dir)))
    tr = pd.read_csv(os.path.join(args.cache_dir, "train_index.csv"))
    va = pd.read_csv(os.path.join(args.cache_dir, "valid_index.csv"))
    if args.merge_valid:
        tr = pd.concat([tr, va], ignore_index=True)
    print(f"train imgs={len(tr)} valid imgs={len(va)} device={device}")

    dl_tr = DataLoader(CXRDataset(tr, args.img_size, True, args.hflip), batch_size=args.bs, shuffle=True,
                       num_workers=args.workers, pin_memory=amp, drop_last=True, persistent_workers=args.workers > 0)
    dl_va = DataLoader(CXRDataset(va, args.img_size, False), batch_size=args.bs * 2, shuffle=False,
                       num_workers=args.workers, pin_memory=amp)

    model = CXRNet(args.backbone, pretrained=not args.no_pretrained, pretrained_path=args.pretrained_path).to(device)
    ema = EMA(model, args.ema) if args.ema > 0 else None
    crit = AsymmetricLoss() if args.loss == "asl" else nn.BCEWithLogitsLoss()
    head_params = list(model.head.parameters()) + [model.p]
    opt = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": args.lr},
                             {"params": head_params, "lr": args.lr * 5}], weight_decay=args.wd)
    steps_ep = len(dl_tr) if not args.max_steps else min(len(dl_tr), args.max_steps)
    total, warm = steps_ep * args.epochs, int(steps_ep * args.warmup)
    base = [g["lr"] for g in opt.param_groups]
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    cfg = dict(backbone=args.backbone, img_size=args.img_size, cache_size=cache_size,
               epochs=args.epochs, agg=args.agg, merge_valid=args.merge_valid, seed=args.seed)
    if args.pretrained_path:
        cfg["pretrained_file"] = os.path.basename(args.pretrained_path)
    os.makedirs(args.out_dir, exist_ok=True)
    best, step, log = -1, 0, []
    for ep in range(args.epochs):
        model.train()
        t0, run = time.time(), 0.0
        for i, (x, y) in enumerate(dl_tr):
            if args.max_steps and i >= args.max_steps:
                break
            f = step / max(1, warm) if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
            for g, b in zip(opt.param_groups, base):
                g["lr"] = b * f
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                loss = crit(model(x), y)
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
        eval_model = model
        if ema:  # 用 EMA 权重评估/保存
            eval_model = CXRNet(args.backbone, pretrained=False).to(device)
            ema.copy_to(eval_model)
        if not args.merge_valid:
            auc, aucs, f1, st, y = evaluate(eval_model, dl_va, va, device, amp, args.agg)
            msg.update(val_auc=round(auc, 4), val_f1_05=round(f1, 4))
            if auc > best:
                best = auc
                save_checkpoint(eval_model, {**cfg, "best_epoch": ep + 1, "val_auc": auc}, args.out_dir)
                st.to_csv(os.path.join(args.out_dir, "valid_study_preds.csv"), index=False)
                np.save(os.path.join(args.out_dir, "valid_study_labels.npy"), y)
                msg["saved"] = True
                msg["per_class_auc"] = [round(a, 4) for a in aucs]
        elif ep == args.epochs - 1:
            save_checkpoint(eval_model, cfg, args.out_dir)
            msg["saved"] = True
        print(msg, flush=True)
        log.append(msg)
        save_json(log, os.path.join(args.out_dir, "train_log.json"))
    print("done. best val macro-AUC =", round(best, 4) if best >= 0 else "n/a (merge_valid)")


if __name__ == "__main__":
    main()
