# -*- coding: utf-8 -*-
"""全流程冒烟测试（无需真实数据/GPU）：
生成与赛方结构一致的合成数据 -> preprocess -> 2 个小模型训练 -> calibrate -> predict -> cost -> manifest。
用法: python smoke_test.py  (约几分钟，CPU 即可)"""
import os
import sys
import shutil
import subprocess


HERE = os.path.dirname(os.path.abspath(__file__))
TMP = "/tmp/cxr_smoke"
DATA = os.path.join(TMP, "cxr_first")
PY = sys.executable


def make_split(split, n_study, start, rng, labeled=True):
    rows, pid = [], start
    for s in range(n_study):
        sub, stu = f"subject_{start + s // 2:05d}", f"study_{start + s:05d}"
        y = rng.random(10) < 0.25
        if not y[:9].any():
            y[:] = False
            y[9] = True
        lab = ";".join(str(k + 1) for k in np.where(y)[0])
        for _ in range(rng.integers(1, 3)):
            did = f"pic_{pid:06d}"
            pid += 1
            h, w = int(rng.integers(300, 420)), int(rng.integers(300, 420))
            img = (rng.random((h, w)) * 60 + 80).astype(np.uint8)
            if y[5]:  # 让“心影扩大”可学：中央亮块
                img[h // 3:2 * h // 3, w // 3:2 * w // 3] = 230
            if y[8]:  # “胸腔积液”：底部亮带
                img[-h // 5:] = 220
            d = os.path.join(DATA, split, sub, stu)
            os.makedirs(d, exist_ok=True)
            Image.fromarray(img).save(os.path.join(d, did + (".jpg" if pid % 3 else ".png")), quality=92)
            r = dict(Dicom_id=did, Subject_id=sub, Study_id=stu, ViewPosition="AP", PatientOrientation="Erect",
                     Rows=h, Columns=w)
            if labeled:
                r["Predict_class"] = lab
            rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(DATA, f"{split}.csv"), index=False)
    return pd.DataFrame(rows)


def run(cmd, cwd):
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def gen():
    shutil.rmtree(TMP, ignore_errors=True)
    rng = np.random.default_rng(0)
    make_split("train", 120, 0, rng)
    make_split("valid", 40, 1000, rng)
    make_split("test", 30, 2000, rng, labeled=False)
    # pictures_100 / pictures_5：从测试集拷贝，保持 subject/study 目录结构
    for name, sl in [("pictures_100", slice(0, 40)), ("pictures_5", slice(40, 45))]:
        for p in sorted(__import__("glob").glob(os.path.join(DATA, "test", "*", "*", "*")))[sl]:
            dst = os.path.join(TMP, name, os.path.relpath(p, os.path.join(DATA, "test")))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(p, dst)


def check_submission():
    test = pd.read_csv(os.path.join(DATA, "test.csv"))
    s = pd.read_csv(os.path.join(TMP, "submission.csv"))
    assert list(s.columns) == ["Subject_id", "Study_id", "Predict_class", "Probability"]
    assert s.Study_id.nunique() == test.Study_id.nunique() and len(s) == 10 * test.Study_id.nunique()
    assert s.Probability.between(0, 1).all()
    print("submission.csv 格式检查通过:", len(s), "行,", s.Study_id.nunique(), "个检查")


def main():
    # 父进程保持轻量（本机仅 1GB 内存），所有重活放子进程
    run([PY, __file__, "--gen"], HERE)
    code, md, cache = os.path.join(HERE, "code"), os.path.join(TMP, "model"), os.path.join(TMP, "cache")
    run([PY, "preprocess.py", "--data_root", DATA, "--cache_dir", cache, "--cache_size", "112", "--workers", "1"], code)
    for bb in ["mobilenetv3_small_050", "test_efficientnet"]:  # 本机 1GB 内存，仅用微型骨干验证流程
        run([PY, "train.py", "--cache_dir", f"{cache}/112", "--out_dir", f"{md}/{bb}", "--backbone", bb,
             "--no_pretrained", "--img_size", "96", "--epochs", "2", "--bs", "4", "--workers", "0",
             "--lr", "1e-3", "--ema", "0.9"], code)
    run([PY, "calibrate.py", "--model_dir", md], code)
    sub = os.path.join(TMP, "submission.csv")
    run([PY, "predict.py", "--data_root", DATA, "--model_dir", md, "--out", sub], code)
    run([PY, __file__, "--check"], HERE)
    run([PY, "cost.py", sub, os.path.join(TMP, "pictures_100"), os.path.join(TMP, "pictures_5"),
         "--model_dir", md], HERE)
    run([PY, "tools.py", "manifest", "--model_dir", md, "--pretrain_dir", os.path.join(TMP, "pretrain_model")], code)
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    if "--gen" in sys.argv or "--check" in sys.argv:
        import numpy as np
        import pandas as pd
        from PIL import Image
        gen() if "--gen" in sys.argv else check_submission()
    else:
        main()
