#!/usr/bin/env bash
# P4 最终版：分辨率 A/B + 全量数据(train+valid)多种子模型（断点续跑：已完成的实验自动跳过）
#  S0 检查 P3 结果；需要 640 时生成 704 缓存 + 与 P3 相同划分的折文件
#  S1 EffV2-S 第0折：576 / 640 与 P3 的 512 比较（只看官方训练集第0折折外 AUC；大分辨率需至少 +MIN_GAIN）
#  S2 选定分辨率 R、最佳轮数 E 后，用 train+valid 训练 NFULL 个不同种子的全量模型（余弦按 12 轮，第 E 轮保存）
#  S3 聚合方式/阈值由同配方 OOF 决定 -> model_p4/；训练集图片测速，超限自动降级（减模型 -> 关 TTA）
#  S4 测试集推理 -> output_p4/；官方验证集训练内自检；文件清单；有 pictures_100/5 时跑 cost.py
# 产物在 exp_p4/ model_p4/ output_p4/，不覆盖 model/ model_p3/ output_p3/
set -uo pipefail
cd "$(dirname "$0")/code"
ROOT=..
mkdir -p $ROOT/exp_p4 $ROOT/logs
PRE=$ROOT/pretrain_model
DATA=$ROOT/../cxr_first
EXP=$ROOT/exp_p4
P3=$ROOT/exp_p3
NF=5
WORKERS=${WORKERS:-8}
LIMIT_MS=${LIMIT_MS:-80}          # 本机测速上限（给主办方硬件留余量）
EPOCHS=${EPOCHS:-12}
RES_LIST=${RES_LIST:-"576 640"}
MIN_GAIN=${MIN_GAIN:-0.002}
NFULL=${NFULL:-3}
SEEDS=(11 22 33 44 55)
EXTRA=${EXTRA:-}
BB=${BB_EFF:-tf_efficientnetv2_s.in21k_ft_in1k}
BS=${BS:-24}
PRE_ARG=""; [ -f "$PRE/$BB.safetensors" ] && PRE_ARG="--pretrained_path $PRE/$BB.safetensors"
cache_of() { if [ "$1" -gt 576 ]; then echo $ROOT/cache/704; else echo $ROOT/cache/576; fi; }
bs_of()    { if [ "$1" -gt 576 ]; then echo $((BS*2/3)); else echo $BS; fi; }   # 640 降 batch 防显存不足
fauc()  { python -c "import json;print(max(r.get('val_auc',-1) for r in json.load(open('$1/train_log.json'))))"; }
fep()   { python -c "import json;l=json.load(open('$1/train_log.json'));print(max(l,key=lambda r:r.get('val_auc',-1))['epoch'])"; }
better(){ python -c "import sys;sys.exit(0 if float('$1')>=float('$2')+float('$3') else 1)"; }   # $1 >= $2+$3 ?

# ---------------- S0
UM=1; [ -f $P3/eff_m1_lr3e-4_f0/DONE ] || UM=0
P3P=eff_m${UM}_lr3e-4
for k in $(seq 0 $((NF-1))); do
  [ -f $P3/${P3P}_f$k/DONE ] || { echo "[ERR] 缺少 P3 结果 $P3/${P3P}_f$k，请先跑完 run_p3.sh"; exit 1; }
done
echo "==> P3 基线：$P3P（use_meta=$UM）"
if [[ " $RES_LIST " == *" 640 "* ]]; then
  C7=$ROOT/cache/704
  if [ ! -f $C7/valid_index.csv ]; then
    echo "[run ] 生成 704 缓存 $(date '+%F %T')"
    python preprocess.py --data_root $DATA --cache_dir $ROOT/cache --cache_size 704 --workers $WORKERS \
      > $ROOT/logs/p4_preprocess704.log 2>&1 || { echo "[FAIL] 见 logs/p4_preprocess704.log"; exit 1; }
  fi
  if [ ! -f $C7/train_folds.csv ]; then
    python - <<PY || exit 1
import pandas as pd
f = pd.read_csv("$ROOT/cache/576/train_folds.csv")[["Dicom_id", "fold"]]
t = pd.read_csv("$C7/train_index.csv").merge(f, on="Dicom_id", how="left")
assert t.fold.notna().all(), "704 缓存与 576 折划分的图像不一致"
t["fold"] = t.fold.astype(int); t.to_csv("$C7/train_folds.csv", index=False)
print("写出 $C7/train_folds.csv（与 P3 相同的患者划分）")
PY
  fi
fi

run() {  # run <tag> <train.py 参数...>
  local tag=$1 out=$EXP/$1; shift
  if [ -f "$out/DONE" ]; then echo "[skip] $tag"; return 0; fi
  mkdir -p "$out"; echo "[run ] $tag   $(date '+%F %T')"
  if python train.py --backbone $BB $PRE_ARG --workers $WORKERS --epochs $EPOCHS --drop_path 0.3 --wd 0.1 \
       --lr 3e-4 --loss asl --hflip 0.5 --use_meta $UM "$@" $EXTRA --out_dir "$out" > $ROOT/logs/p4_$tag.log 2>&1; then
    touch "$out/DONE"
  else echo "[FAIL] $tag，见 logs/p4_$tag.log"; tail -5 $ROOT/logs/p4_$tag.log; exit 1; fi
}

# ---------------- S1 分辨率 A/B（官方训练集第0折）
R=512; BEST=$(fauc $P3/${P3P}_f0)
echo "==> 512 第0折 AUC=$BEST（P3）"
for r in $RES_LIST; do
  C=$(cache_of $r)
  run eff_r${r}_f0 --cache_dir $C --folds_csv $C/train_folds.csv --fold 0 --img_size $r --bs $(bs_of $r) --seed 7
  A=$(fauc $EXP/eff_r${r}_f0)
  grep holdout $ROOT/logs/p4_eff_r${r}_f0.log | sed 's/^/       -> /'
  if better $A $BEST $MIN_GAIN; then echo "==> $r 第0折 AUC=$A ≥ $BEST+$MIN_GAIN，采用 $r"; R=$r; BEST=$A
  else echo "==> $r 第0折 AUC=$A，未超过 $BEST+$MIN_GAIN，不采用"; fi
done
if [ $R -eq 512 ]; then
  OOF=""; EPS=""
  for k in $(seq 0 $((NF-1))); do OOF="$OOF $P3/${P3P}_f$k"; EPS="$EPS,$(fep $P3/${P3P}_f$k)"; done
  E=$(python -c "import statistics;print(int(statistics.median([${EPS#,}])))")
else
  OOF="$EXP/eff_r${R}_f0"; E=$(fep $EXP/eff_r${R}_f0)
fi
echo "==> 最终分辨率=$R  全量训练第 $E 轮保存（余弦计划按 $EPOCHS 轮）  阈值来源:$(for d in $OOF; do printf ' %s' $(basename $d); done)"

# ---------------- S2 全量模型（train+valid）
C=$(cache_of $R); FULL=()
for i in $(seq 0 $((NFULL-1))); do
  s=${SEEDS[$i]}
  run full_r${R}_s$s --cache_dir $C --merge_valid --stop_epoch $E --img_size $R --bs $(bs_of $R) --seed $s
  python -c "import json;l=json.load(open('$EXP/full_r${R}_s$s/train_log.json'));print(f'       -> 第{l[-1][\"epoch\"]}轮保存 loss={l[-1][\"loss\"]:.4f}')"
  FULL+=("$EXP/full_r${R}_s$s")
done

# ---------------- S3 标定 + 测速降级（测速用训练集图片，不用测试集做任何决定）
BENCH=$ROOT/bench_train
if [ ! -d $BENCH/p100 ]; then
  python - <<PY
import os, shutil, pandas as pd
df = pd.read_csv("$ROOT/cache/576/train_index.csv").sample(105, random_state=0)
for i, r in enumerate(df.itertuples()):
    d = os.path.join("$BENCH", "p100" if i < 100 else "p5", r.Subject_id, r.Study_id)
    os.makedirs(d, exist_ok=True); shutil.copy(r.path, d)
PY
fi
calib() { python calibrate_full.py --oof_dirs $OOF --model_dirs "${FULL[@]:0:$1}" --out_dir $ROOT/model_p4 --tta $2 \
            > $ROOT/logs/p4_calibrate.log 2>&1 || { cat $ROOT/logs/p4_calibrate.log; exit 1; }; }
bench() { python $ROOT/cost.py none $BENCH/p100 $BENCH/p5 --model_dir $ROOT/model_p4 --no_check \
            --json_out $ROOT/logs/p4_bench.json > $ROOT/logs/p4_bench.log 2>&1 || { tail -20 $ROOT/logs/p4_bench.log >&2; exit 1; }
          python -c "import json;print(round(json.load(open('$ROOT/logs/p4_bench.json'))['avg_ms'],1))"; }
PLAN=(); for n in $(seq $NFULL -1 2); do PLAN+=("$n:1"); done
if [ $NFULL -ge 2 ]; then PLAN+=("2:0"); fi
PLAN+=("1:1" "1:0")
MS=0
for p in "${PLAN[@]}"; do
  n=${p%%:*}; t=${p##*:}
  calib $n $t; MS=$(bench) || exit 1
  echo "==> 测速(训练集图片)：$n 个全量模型 TTA=$t -> ${MS} ms"
  better $MS $LIMIT_MS 0.0001 || break
done
better $MS $LIMIT_MS 0.0001 && echo "[WARN] 最省配置仍超过 ${LIMIT_MS}ms，请在比赛硬件上用 cost.py 复核"
cat $ROOT/logs/p4_calibrate.log

# ---------------- S4 推理 + 自检 + 清单 + 核验
mkdir -p $ROOT/output_p4/selfcheck
python predict.py --data_root $DATA --model_dir $ROOT/model_p4 --out $ROOT/output_p4/submission.csv || exit 1
python predict.py --data_root $DATA --split valid --model_dir $ROOT/model_p4 --out $ROOT/output_p4/selfcheck/valid_pred.csv \
  > $ROOT/logs/p4_selfcheck.log 2>&1 || { tail -20 $ROOT/logs/p4_selfcheck.log; exit 1; }
python - <<PY
import pandas as pd, numpy as np
from utils import parse_labels, macro_auc, NUM_CLASSES
v = pd.read_csv("$DATA/valid.csv").groupby("Study_id", sort=False).Predict_class.first()
s = pd.read_csv("$ROOT/output_p4/selfcheck/valid_pred.csv").pivot(index="Study_id", columns="Predict_class", values="Probability")
s = s.reindex(v.index)[list(range(1, NUM_CLASSES + 1))].values
y = np.stack([parse_labels(x) for x in v.values])
print(f"[自检] 官方验证集 macro-AUC={macro_auc(y, s)[0]:.4f}（验证集已参与全量训练，偏高属正常，只用来确认流程无误；低于 0.81 说明有问题）")
PY
python tools.py manifest --model_dir $ROOT/model_p4 --pretrain_dir $PRE
cd $ROOT
if [ -d pictures_100 ] && [ -d pictures_5 ]; then
  python cost.py output_p4/submission.csv pictures_100 pictures_5 --model_dir model_p4
else
  echo "[INFO] 未找到 pictures_100/pictures_5，耗时与一致性核验请手动运行 cost.py"
fi
echo "P4 完成 $(date '+%F %T')：output_p4/submission.csv（分辨率 $R，第 $E 轮保存）"
