# -*- coding: utf-8 -*-
"""P4 冒烟测试（无需真实数据/GPU）：先跑 smoke_test_p3.py 产出 exp_p3，再在同一合成项目上跑 run_p4.sh。
LIMIT_MS=1 故意让每档测速都超限，覆盖完整降级路径。用法: python smoke_test_p4.py"""
import os
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = "/tmp/cxr_smoke/proj/cxr_competition"
PY = sys.executable


def run(cmd, cwd, env=None):
    print("\n$", cmd, flush=True)
    subprocess.run(cmd, cwd=cwd, shell=True, check=True, env=env)


def main():
    run(f"{PY} smoke_test_p3.py", HERE)
    env = dict(os.environ, BB_EFF="mobilenetv3_small_050", EPOCHS="2", WORKERS="0", LIMIT_MS="1", NFULL="2",
               EXTRA="--no_pretrained --img_size 96 --bs 4 --workers 0 --ema 0.9 --max_steps 4 --lr 1e-3")
    run("bash run_p4.sh 2>&1 | grep -v 'it/s'", PROJ, env)
    run(f"{PY} smoke_test.py --check --sub {PROJ}/output_p4/submission.csv --test {PROJ}/../cxr_first/test.csv", HERE)
    print("\nP4 SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
