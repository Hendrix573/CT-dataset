# -*- coding: utf-8 -*-
"""P2+P3 冒烟测试（无需真实数据/GPU）：合成数据 -> 缓存 -> run_p3.sh（微型骨干、每折几步）-> 一致性核验。
用法: python smoke_test_p3.py"""
import os
import sys
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
TMP = "/tmp/cxr_smoke"
PROJ = os.path.join(TMP, "proj")
PY = sys.executable


def run(cmd, cwd, env=None):
    print("\n$", cmd, flush=True)
    subprocess.run(cmd, cwd=cwd, shell=True, check=True, env=env)


def main():
    run(f"{PY} smoke_test.py --gen", HERE)                      # 生成 /tmp/cxr_smoke/cxr_first 等
    shutil.rmtree(PROJ, ignore_errors=True)
    os.makedirs(PROJ)
    shutil.copytree(HERE, os.path.join(PROJ, "cxr_competition"),
                    ignore=lambda d, names: [n for n in names if os.path.abspath(d) == HERE and
                                             (n in ("cache", "logs", "bench_train") or n.startswith(("exp", "model", "output")))])
    shutil.move(os.path.join(TMP, "cxr_first"), os.path.join(PROJ, "cxr_first"))
    for d in ["pictures_100", "pictures_5"]:
        shutil.move(os.path.join(TMP, d), os.path.join(PROJ, "cxr_competition", d))
    # 随机化视角元数据，覆盖 PA/AP/LATERAL/LL/空值
    run(f"""{PY} -c "
import pandas as pd, numpy as np
rng=np.random.default_rng(1)
for s in ['train','valid','test']:
    p='{PROJ}/cxr_first/'+s+'.csv'; d=pd.read_csv(p)
    d['ViewPosition']=rng.choice(['PA','AP','LATERAL','LL',None],len(d))
    d['PatientOrientation']=rng.choice(['Erect','Recumbent',None],len(d))
    d.to_csv(p,index=False)
" """, HERE)
    code = os.path.join(PROJ, "cxr_competition", "code")
    run(f"{PY} preprocess.py --data_root ../../cxr_first --cache_dir ../cache --cache_size 576 --workers 1", code)
    env = dict(os.environ, BB_EFF="mobilenetv3_small_050", BB_CNXS="test_efficientnet", EPOCHS="1",
               WORKERS="0", LIMIT_MS="10000",
               EXTRA="--no_pretrained --img_size 96 --bs 4 --workers 0 --ema 0.9 --max_steps 4 --lr 1e-3")
    run("bash run_p3.sh 2>&1 | grep -v 'it/s'", os.path.join(PROJ, "cxr_competition"), env)
    run(f"{PY} smoke_test.py --check --sub {PROJ}/cxr_competition/output_p3/submission.csv "
        f"--test {PROJ}/cxr_first/test.csv", HERE)
    print("\nP3 SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
