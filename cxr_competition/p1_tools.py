# -*- coding: utf-8 -*-
"""P1 调参辅助：读取 exp/*/train_log.json + params.txt
  python p1_tools.py show <tag> | best <arch> | table | subset <model_dir>"""
import os, sys, json, glob, itertools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "exp")

def load(tag):
    d = os.path.join(EXP, tag)
    log = json.load(open(os.path.join(d, "train_log.json"), encoding="utf-8"))
    params = open(os.path.join(d, "params.txt")).read().split()
    aucs = [r.get("val_auc", -1) for r in log]
    i = int(np.argmax(aucs))
    return dict(tag=tag, params=params, auc=aucs[i], best_ep=i + 1, n_ep=len(log),
                done=os.path.exists(os.path.join(d, "DONE")))

def runs(prefix=""):
    out = []
    for d in sorted(glob.glob(os.path.join(EXP, prefix + "*"))):
        try:
            r = load(os.path.basename(d))
            if r["done"]:
                out.append(r)
        except Exception:
            pass
    return out

def main():
    cmd = sys.argv[1]
    if cmd == "show":
        r = load(sys.argv[2])
        print(f"       -> best val_auc={r['auc']:.4f} @ epoch {r['best_ep']}/{r['n_ep']}")
    elif cmd == "best":
        rs = runs(sys.argv[2] + "_")
        if not rs:
            sys.exit(f"no finished runs for {sys.argv[2]}")
        r = max(rs, key=lambda r: r["auc"])
        print(" ".join(r["params"]), r["tag"], f"{r['auc']:.4f}", r["best_ep"])
    elif cmd == "table":
        for r in sorted(runs(), key=lambda r: -r["auc"]):
            print(f"{r['auc']:.4f}  best_ep={r['best_ep']:>2}/{r['n_ep']:<2}  {r['tag']}")
    elif cmd == "subset":
        sys.path.insert(0, os.path.join(HERE, "code"))
        import pandas as pd
        from utils import macro_auc
        md = sys.argv[2]
        names = sorted(n for n in os.listdir(md) if os.path.exists(os.path.join(md, n, "valid_study_preds.csv")))
        pcols = [f"p{k}" for k in range(1, 11)]
        P = {n: pd.read_csv(os.path.join(md, n, "valid_study_preds.csv"))[pcols].values for n in names}
        y = np.load(os.path.join(md, names[0], "valid_study_labels.npy"))
        full = macro_auc(y, np.mean([P[n] for n in names], 0))[0]
        best = (full, tuple(names))
        print(f"  全部 {names}: {full:.4f}")
        for k in range(1, len(names)):
            for c in itertools.combinations(names, k):
                a = macro_auc(y, np.mean([P[n] for n in c], 0))[0]
                print(f"  {list(c)}: {a:.4f}")
                if a > best[0] + 0.001:   # 子集须明显优于全集才采用，避免对验证集过拟合
                    best = (a, c)
        print("KEEP", " ".join(best[1]))

if __name__ == "__main__":
    main()