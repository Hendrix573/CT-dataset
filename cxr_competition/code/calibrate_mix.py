# -*- coding: utf-8 -*-
"""P5：EffV2-S + EffV2-M 异构集成的门槛判断与标定。所有选择只用官方训练集 OOF；官方验证集只做留出报告。
子命令：
  gate  --a <S 第0折目录> --b <M 第0折目录> [--json f]   第0折折外 S / M / S+M 等权 的检查级 macro-AUC
  cv    --group eff=<前缀> --group effm=<前缀> --exp_dirs <目录...> [--json f]   5折 OOF 与官方验证集留出
  final --group ... --exp_dirs ... --models <全量模型目录...> --out_dir ../model_p5 --tta 1
等权平均：Predictor 对所有模型等权平均，因此最终 S 与 M 全量模型数量必须相同（run_p5.sh 保证）。"""
import os
import json
import shutil
import argparse

import numpy as np
import pandas as pd

from utils import PCOLS, NUM_CLASSES, CLASS_NAMES, study_labels, macro_auc, macro_f1, best_thresholds, remap_probs, save_json
from calibrate_cv import AGGS, score

FIX = ("mean", 2.0)   # P3/P4 选出的聚合方式，门槛比较时固定使用


def avg(frames):
    ids = frames[0].Dicom_id.values
    for f in frames[1:]:
        assert np.array_equal(f.Dicom_id.values, ids), "各组图像集合不一致"
    out = frames[0].copy()
    out[PCOLS] = np.mean([f[PCOLS].values for f in frames], 0)
    return out


def read_sorted(path):
    return pd.read_csv(path).sort_values("Dicom_id").reset_index(drop=True)


def find_dir(exp_dirs, name):
    for e in exp_dirs:
        d = os.path.join(e, name)
        if os.path.exists(os.path.join(d, "DONE")):
            return d
    raise SystemExit(f"找不到已完成的实验 {name}（搜索 {exp_dirs}）")


def load_group(exp_dirs, prefix, nfolds):
    dirs = [find_dir(exp_dirs, f"{prefix}_f{k}") for k in range(nfolds)]
    oof = pd.concat([pd.read_csv(os.path.join(d, "oof_img_preds.csv")) for d in dirs], ignore_index=True)
    assert oof.Dicom_id.is_unique, f"{prefix}: OOF 图像重复"
    hos = [read_sorted(os.path.join(d, "holdout_img_preds.csv")) for d in dirs]
    return oof.sort_values("Dicom_id").reset_index(drop=True), avg(hos)


def dump(path, obj):
    if path:
        with open(path, "w") as f:
            json.dump(obj, f)


def cmd_gate(a):
    A = read_sorted(os.path.join(a.a, "oof_img_preds.csv"))
    B = read_sorted(os.path.join(a.b, "oof_img_preds.csv"))
    sa, sb, se = score(A, *FIX)[0], score(B, *FIX)[0], score(avg([A, B]), *FIX)[0]
    print(f"[gate] 第0折折外 macro-AUC：S={sa:.4f}  M={sb:.4f}  S+M={se:.4f}  集成增益={se - sa:+.4f}")
    dump(a.json, dict(s=sa, m=sb, ens=se, gain=se - sa))


def groups_data(a):
    g = dict(x.split("=", 1) for x in a.group)
    return {n: load_group(a.exp_dirs, p, a.nfolds) for n, p in g.items()}


def cmd_cv(a):
    data = groups_data(a)
    res = {}
    for n, (oof, ho) in data.items():
        res[n] = (score(oof, *FIX)[0], score(ho, *FIX)[0])
        print(f"[cv] {n:6s} OOF={res[n][0]:.4f}  官方验证集(5折平均)={res[n][1]:.4f}")
    names = list(data)
    eo = score(avg([data[n][0] for n in names]), *FIX)[0]
    eh = score(avg([data[n][1] for n in names]), *FIX)[0]
    b = names[0]
    print(f"[cv] {'+'.join(names)} OOF={eo:.4f}（增益 {eo - res[b][0]:+.4f}）  "
          f"官方验证集={eh:.4f}（增益 {eh - res[b][1]:+.4f}）")
    dump(a.json, dict(oof=eo, holdout=eh, oof_gain=eo - res[b][0], holdout_gain=eh - res[b][1]))


def cmd_final(a):
    data = groups_data(a)
    oof = avg([d[0] for d in data.values()])
    ho = avg([d[1] for d in data.values()])
    auc, mode, w = max((score(oof, m, w)[0], m, w) for m, w in AGGS)
    _, st = score(oof, mode, w)
    y = study_labels(oof)
    ths = best_thresholds(y, st[PCOLS].values)
    f1_opt = macro_f1(y, remap_probs(st[PCOLS].values, ths))[0]
    ho_auc, ho_st = score(ho, mode, w)
    _, ho_aucs = macro_auc(study_labels(ho), ho_st[PCOLS].values)
    print(f"OOF：聚合={mode} 正位权重={w:g}  macro-AUC={auc:.4f}  重映射后 macro-F1={f1_opt:.4f}")
    print(f"官方验证集（5折模型集成，留出）：macro-AUC={ho_auc:.4f}")
    for k in range(NUM_CLASSES):
        print(f"  {k+1:2d} {CLASS_NAMES[k]:28s} 验证集AUC={ho_aucs[k]:.4f} thr={ths[k]:.2f}")
    if os.path.isdir(a.out_dir):
        shutil.rmtree(a.out_dir)
    os.makedirs(a.out_dir)
    models = []
    for d in a.models:
        dst = os.path.join(a.out_dir, os.path.basename(os.path.normpath(d)))
        os.makedirs(dst)
        for f in ["best.pth", "cfg.json", "train_log.json"]:
            if os.path.exists(os.path.join(d, f)):
                shutil.copy(os.path.join(d, f), dst)
        models.append(os.path.basename(dst))
    save_json(dict(thresholds=ths, agg=mode, frontal_w=w, tta=bool(a.tta), models=models,
                   oof_macro_auc=auc, oof_macro_f1=f1_opt, holdout_macro_auc_cv=ho_auc,
                   note="EffV2-S 与 EffV2-M 全量模型等量等权；轮数、聚合、阈值均在合并前由官方训练集 OOF 决定"),
              os.path.join(a.out_dir, "thresholds.json"))
    print(f"已写入 {a.out_dir}：{len(models)} 个模型 + thresholds.json (tta={bool(a.tta)})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate")
    g.add_argument("--a", required=True)
    g.add_argument("--b", required=True)
    g.add_argument("--json")
    for name in ["cv", "final"]:
        p = sub.add_parser(name)
        p.add_argument("--group", action="append", required=True)
        p.add_argument("--exp_dirs", nargs="+", required=True)
        p.add_argument("--nfolds", type=int, default=5)
        p.add_argument("--json")
        if name == "final":
            p.add_argument("--models", nargs="+", required=True)
            p.add_argument("--out_dir", default="../model_p5")
            p.add_argument("--tta", type=int, default=1)
    a = ap.parse_args()
    {"gate": cmd_gate, "cv": cmd_cv, "final": cmd_final}[a.cmd](a)


if __name__ == "__main__":
    main()
