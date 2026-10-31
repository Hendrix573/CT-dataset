#!/usr/bin/env bash
# P5：EfficientNetV2-M 异构集成（断点续跑：已完成的实验自动跳过）
#  S0 检查 P3 五折 / P4 全量模型 / EffV2-M 预训练权重
#  S1 EffV2-M 第0折：lr 2e-4 / 3e-4；与 P3 的 EffV2-S 第0折等权集成，折外增益 < GATE1 则停止
#  S2 EffV2-M 跑满 5 折；5 折 OOF 集成增益 < GATE2 则停止（官方验证集只做留出报告）
#  S3 EffV2-M 全量模型（train+valid，NFULL 个种子，第 E 轮保存，E = 5 折最佳轮中位数）
#  S4 S、M 全量模型等量等权；聚合/阈值由 5 折 OOF 决定 -> model_p5/；训练集图片测速，超限降级
#  S5 测试集推理 -> output_p5/；训练内自检；文件清单；有 pictures_100/5 时跑 cost.py
# 产物在 exp_p5/ model_p5/ output_p5/，不覆盖 model/ model_p3/ model_p4/
set -uo pipefail
cd "$(dirname "$0")/code"
ROOT=..
mkdir -p $ROOT/exp_p5 $ROOT/logs
PRE=$ROOT/pretrain_model
DATA=$ROOT/../cxr_first
CACHE=$ROOT/cache/576
EXP=$ROOT/exp_p5
P3=$ROOT/exp_p3
P4=$ROOT/exp_p4
NF=5
WORKERS=${WORKERS:-8}
LIMIT_MS=${LIMIT_MS:-80}
EPOCHS=${EPOCHS:-12}
LR_LIST=${LR_LIST:-"2e-4 3e-4"}
GATE1=${GATE1:-0.003}
GATE2=${GATE2:-0.002}
NFULL=${NFULL:-2}
SEEDS=(11 22 33)
EXTRA=${EXTRA:-}
BB=${BB_M:-tf_efficientnetv2_m.in21k_ft_in1k}
BS=${BS_M:-16}
fep()   { python -c "import json;l=json.load(open('$1/train_log.json'));print(max(l,key=lambda r:r.get('val_auc',-1))['epoch'])"; }
better(){ python -c "import sys;sys.exit(0 if float('$1')>=float('$2')+float('$3') else 1)"; }   # $1 >= $2+$3 ?
jget()  { python -c "import json;print(round(json.load(open('$1'))['$2'],5))"; }

# ---------------- S0
UM=1; [ -f $P3/eff_m1_lr3e-4_f0/DONE ] || UM=0
SP=eff_m${UM}_lr3e-4
for k in $(seq 0 $((NF-1))); do
  [ -f $P3/${SP}_f$k/DONE ] || { echo "[ERR] 缺少 P3 结果 $P3/${SP}_f$k"; exit 1; }
done
SFULL=()
for d in $P4/full_r512_s*; do [ -f "$d/DONE" ] && SFULL+=("$d"); done
[ ${#SFULL[@]} -ge $NFULL ] || { echo "[ERR] exp_p4 里 512 全量 EffV2-S 只有 ${#SFULL[@]} 个，需要 ≥ $NFULL"; exit 1; }
PRE_ARG=""
if [ -f "$PRE/$BB.safetensors" ]; then PRE_ARG="--pretrained_path $PRE/$BB.safetensors"
elif [[ "$EXTRA" != *"--no_pretrained"* ]]; then
  echo "[ERR] 缺少 $PRE/$BB.safetensors"
  echo "      下载 https://hf-mirror.com/timm/$BB/resolve/main/model.safetensors 并改成上面的文件名"; exit 1
fi
echo "==> S 基线：$SP（5折）+ 全量 $(for d in "${SFULL[@]}"; do printf '%s ' $(basename $d); done)；M 骨干：$BB"

run() {  # run <tag> <train.py 参数...>
  local tag=$1 out=$EXP/$1; shift
  if [ -f "$out/DONE" ]; then echo "[skip] $tag"; return 0; fi
  mkdir -p "$out"; echo "[run ] $tag   $(date '+%F %T')"
  if python train.py --backbone $BB $PRE_ARG --cache_dir $CACHE --workers $WORKERS --epochs $EPOCHS --img_size 512 \
       --bs $BS --drop_path 0.3 --wd 0.1 --loss asl --hflip 0.5 --use_meta $UM "$@" $EXTRA --out_dir "$out" \
       > $ROOT/logs/p5_$tag.log 2>&1; then
    touch "$out/DONE"
    if [ -f "$out/oof_img_preds.csv" ]; then
      python -c "import json;l=json.load(open('$out/train_log.json'));b=max(l,key=lambda r:r.get('val_auc',-1));print(f\"       -> best={b['val_auc']:.4f} @ep{b['epoch']}/{len(l)}\")"
      grep holdout $ROOT/logs/p5_$tag.log | sed 's/^/       -> /'
    fi
  else echo "[FAIL] $tag，见 logs/p5_$tag.log"; tail -5 $ROOT/logs/p5_$tag.log; exit 1; fi
}

# ---------------- S1 第0折学习率 + 门槛
LR=""; GBEST=-1
for lr in $LR_LIST; do
  run effm_lr${lr}_f0 --folds_csv $CACHE/train_folds.csv --fold 0 --lr $lr --seed 7
  python calibrate_mix.py gate --a $P3/${SP}_f0 --b $EXP/effm_lr${lr}_f0 --json $ROOT/logs/p5_gate_$lr.json || exit 1
  g=$(jget $ROOT/logs/p5_gate_$lr.json gain)
  if better $g $GBEST 0.00001; then GBEST=$g; LR=$lr; fi
done
echo "==> 第0折：选 lr=$LR，S+M 集成增益=$GBEST（门槛 $GATE1）"
if ! better $GBEST 0 $GATE1; then
  echo "==> 增益未达门槛，P5 停止。EffV2-M 带来的差异度不够，最终版保持 P4，不建议花提交机会。"; exit 0
fi

# ---------------- S2 满 5 折 + 门槛
for k in $(seq 1 $((NF-1))); do run effm_lr${LR}_f$k --folds_csv $CACHE/train_folds.csv --fold $k --lr $LR --seed 7; done
G="--group eff=$SP --group effm=effm_lr$LR --exp_dirs $P3 $EXP --nfolds $NF"
python calibrate_mix.py cv $G --json $ROOT/logs/p5_cv.json | tee $ROOT/logs/p5_cv.log; [ ${PIPESTATUS[0]} -eq 0 ] || exit 1
OG=$(jget $ROOT/logs/p5_cv.json oof_gain); HG=$(jget $ROOT/logs/p5_cv.json holdout_gain)
if ! better $OG 0 $GATE2; then
  echo "==> 5折 OOF 集成增益 $OG < $GATE2，P5 停止，最终版保持 P4。"; exit 0
fi
better $HG 0 0 || echo "[WARN] 官方验证集留出增益 $HG ≤ 0，和 OOF 方向不一致；脚本继续，但提交前要有心理准备"

# ---------------- S3 EffV2-M 全量模型
EPS=""; for k in $(seq 0 $((NF-1))); do EPS="$EPS,$(fep $EXP/effm_lr${LR}_f$k)"; done
E=$(python -c "import statistics;print(int(statistics.median([${EPS#,}])))")
echo "==> EffV2-M 全量训练：lr=$LR，第 $E 轮保存（5折最佳轮:${EPS#,}）"
MFULL=()
for i in $(seq 0 $((NFULL-1))); do
  s=${SEEDS[$i]}
  run fullm_s$s --merge_valid --stop_epoch $E --lr $LR --seed $s
  python -c "import json;l=json.load(open('$EXP/fullm_s$s/train_log.json'));print(f'       -> 第{l[-1][\"epoch\"]}轮保存 loss={l[-1][\"loss\"]:.4f}')"
  MFULL+=("$EXP/fullm_s$s")
done

# ---------------- S4 标定 + 测速降级（S、M 等量；测速只用训练集图片）
BENCH=$ROOT/bench_train
if [ ! -d $BENCH/p100 ]; then
  python - <<PY
import os, shutil, pandas as pd
df = pd.read_csv("$CACHE/train_index.csv").sample(105, random_state=0)
for i, r in enumerate(df.itertuples()):
    d = os.path.join("$BENCH", "p100" if i < 100 else "p5", r.Subject_id, r.Study_id)
    os.makedirs(d, exist_ok=True); shutil.copy(r.path, d)
PY
fi
calib() { python calibrate_mix.py final $G --models "${SFULL[@]:0:$1}" "${MFULL[@]:0:$1}" --out_dir $ROOT/model_p5 --tta $2 \
            > $ROOT/logs/p5_calibrate.log 2>&1 || { cat $ROOT/logs/p5_calibrate.log; exit 1; }; }
bench() { python $ROOT/cost.py none $BENCH/p100 $BENCH/p5 --model_dir $ROOT/model_p5 --no_check \
            --json_out $ROOT/logs/p5_bench.json > $ROOT/logs/p5_bench.log 2>&1 || { tail -20 $ROOT/logs/p5_bench.log >&2; exit 1; }
          python -c "import json;print(round(json.load(open('$ROOT/logs/p5_bench.json'))['avg_ms'],1))"; }
PLAN=(); for n in $(seq $NFULL -1 1); do PLAN+=("$n:1" "$n:0"); done
MS=0
for p in "${PLAN[@]}"; do
  n=${p%%:*}; t=${p##*:}
  calib $n $t; MS=$(bench) || exit 1
  echo "==> 测速(训练集图片)：EffV2-S×$n + EffV2-M×$n，TTA=$t -> ${MS} ms"
  better $MS $LIMIT_MS 0.0001 || break
done
if better $MS $LIMIT_MS 0.0001; then
  echo "[ERR] S×1+M×1 关 TTA 仍超过 ${LIMIT_MS}ms，P5 不可用，最终版保持 P4"; exit 1
fi
cat $ROOT/logs/p5_calibrate.log

# ---------------- S5 推理 + 自检 + 清单 + 核验
mkdir -p $ROOT/output_p5/selfcheck
python predict.py --data_root $DATA --model_dir $ROOT/model_p5 --out $ROOT/output_p5/submission.csv || exit 1
python predict.py --data_root $DATA --split valid --model_dir $ROOT/model_p5 --out $ROOT/output_p5/selfcheck/valid_pred.csv \
  > $ROOT/logs/p5_selfcheck.log 2>&1 || { tail -20 $ROOT/logs/p5_selfcheck.log; exit 1; }
python - <<PY
import pandas as pd, numpy as np
from utils import parse_labels, macro_auc, NUM_CLASSES
v = pd.read_csv("$DATA/valid.csv").groupby("Study_id", sort=False).Predict_class.first()
s = pd.read_csv("$ROOT/output_p5/selfcheck/valid_pred.csv").pivot(index="Study_id", columns="Predict_class", values="Probability")
s = s.reindex(v.index)[list(range(1, NUM_CLASSES + 1))].values
y = np.stack([parse_labels(x) for x in v.values])
print(f"[自检] 官方验证集 macro-AUC={macro_auc(y, s)[0]:.4f}（验证集已参与全量训练，偏高正常；只确认流程无误，P4 为 0.8676）")
PY
python tools.py manifest --model_dir $ROOT/model_p5 --pretrain_dir $PRE
cd $ROOT
if [ -d pictures_100 ] && [ -d pictures_5 ]; then
  python cost.py output_p5/submission.csv pictures_100 pictures_5 --model_dir model_p5
else
  echo "[INFO] 未找到 pictures_100/pictures_5，耗时与一致性核验请手动运行 cost.py"
fi
echo "P5 完成 $(date '+%F %T')：output_p5/submission.csv（EffV2-M lr=$LR，第 $E 轮）"
