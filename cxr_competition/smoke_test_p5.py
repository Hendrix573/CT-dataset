# -*- coding: utf-8 -*-
"""P5 冒烟测试（无需真实数据/GPU）：smoke_test_p3 -> run_p4(只用512) -> run_p5（门槛设为负数，强制走完全流程）。
用法: python smoke_test_p5.py"""
import os
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = "/tmp/cxr_smoke/proj/cxr_competition"
PY = sys.executable
TINY = "--no_pretrained --img_size 96 --bs 4 --workers 0 --ema 0.9 --max_steps 4 --lr 1e-3"


def run(cmd, cwd, env=None):
    print("\n$", cmd, flush=True)
    subprocess.run(cmd, cwd=cwd, shell=True, check=True, env=env)


def main():
    run(f"{PY} smoke_test_p3.py", HERE)
    for f in ["run_p4.sh", "run_p5.sh", "code/calibrate_full.py", "code/calibrate_mix.py", "code/train.py"]:
        run(f"cp {f} {PROJ}/{f}", HERE)
    base = dict(os.environ, BB_EFF="mobilenetv3_small_050", EPOCHS="2", WORKERS="0", NFULL="2", EXTRA=TINY)
    run("bash run_p4.sh 2>&1 | grep -v 'it/s'", PROJ, dict(base, RES_LIST=" ", LIMIT_MS="10000"))
    env = dict(base, BB_M="test_efficientnet", GATE1="-1", GATE2="-1", LIMIT_MS="1", LR_LIST="2e-4 3e-4")
    run("bash run_p5.sh 2>&1 | grep -v 'it/s'", PROJ, env)
    run(f"{PY} smoke_test.py --check --sub {PROJ}/output_p5/submission.csv --test {PROJ}/../cxr_first/test.csv", HERE)
    print("\nP5 SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
