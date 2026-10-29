#!/usr/bin/env bash
# P2 + P3：视角元数据 + 按患者 5 折交叉验证集成（断点续跑：已完成的折自动跳过）
#  S0 按患者分 5 折（只分官方训练集；官方验证集全程只做留出评估）
#  S1 EffV2-S 第0折：元数据 开/关 A/B
#  S2 ConvNeXt-S 第0折：lr 2e-4 / 3e-4（P1 中 ConvNeXt 欠拟合）
#  S3 两个骨干用选定配置跑满 5 折
#  S4 OOF 选组合/聚合/阈值 -> model_p3/；训练集图片测速，超 LIMIT_MS 自动降级（关TTA -> 只用EffV2）
#  S5 测试集推理 -> output_p3/；文件清单；若有 pictures_100/5 则跑 cost.py 核验
# 产物在 exp_p3/ model_p3/ output_p3/，不覆盖已有 model/ model_p1/ submission.csv
set -uo pipefail
cd "$(dirname "$0")/code"
ROOT=..
mkdir -p $ROOT/exp_p3 $ROOT/logs
CACHE=$ROOT/cache/576
PRE=$ROOT/pretrain_model
DATA=$ROOT/../cxr_first
EXP=$ROOT/exp_p3
NF=5
WORKERS=${WORKERS:-8}
LIMIT_MS=${LIMIT_MS:-80}          # 本机测速上限（留余量给主办方硬件）
EPOCHS=${EPOCHS:-12}
EXTRA=${EXTRA:-}                  # 冒烟测试用：额外传给 train.py 的参数
declare -A BB=( [eff]=tf_efficientnetv2_s.in21k_ft_in1k [cnxs]=convnext_small.fb_in22k_ft_in1k )
declare -A BS=( [eff]=24 [cnxs]=24 )
[ -n "${BB_EFF:-}" ] && BB[eff]=$BB_EFF
[ -n "${BB_CNXS:-}" ] && BB[cnxs]=$BB_CNXS

fold() {  # fold <arch> <lr> <use_meta> <k>
  local a=$1 lr=$2 um=$3 k=$4
  local tag=${a}_m${um}_lr${lr}_f${k} out=$EXP/${a}_m${um}_lr${lr}_f${k}
  if [ -f "$out/DONE" ]; then echo "[skip] $tag"; return 0; fi
  mkdir -p "$out"
  local pre_arg=""
  [ -f "$PRE/${BB[$a]}.safetensors" ] && pre_arg="--pretrained_path $PRE/${BB[$a]}.safetensors"
  echo "[run ] $tag   $(date '+%F %T')"
  if python train.py --backbone ${BB[$a]} $pre_arg --cache_dir $CACHE \
       --folds_csv $CACHE/train_folds.csv --fold $k --img_size 512 --bs ${BS[$a]} --workers $WORKERS \
       --epochs $EPOCHS --lr $lr --drop_path 0.3 --wd 0.1 --loss asl --hflip 0.5 --use_meta $um --seed 7 \
       $EXTRA --out_dir "$out" > $ROOT/logs/p3_$tag.log 2>&1; then
    touch "$out/DONE"
    python -c "import json;l=json.load(open('$out/train_log.json'));b=max(l,key=lambda r:r.get('val_auc',-1));print(f\"       -> 第{$k}折 best={b['val_auc']:.4f} @ep{b['epoch']}/{len(l)}\")"
    grep "holdout" $ROOT/logs/p3_$tag.log | sed 's/^/       -> /'
  else
    echo "[FAIL] $tag，见 logs/p3_$tag.log"; return 1
  fi
}
fauc() { python -c "import json;print(max(r.get('val_auc',-1) for r in json.load(open('$EXP/$1/train_log.json'))))"; }
better() { python -c "import sys;sys.exit(0 if float('$1')>=float('$2')+float('$3') else 1)"; }   # $1 >= $2+$3 ?

# ---------------- S0
[ -f $CACHE/train_folds.csv ] || python make_folds.py --cache_dir $CACHE --nfolds $NF

# ---------------- S1 元数据 A/B（EffV2-S, lr 3e-4）
fold eff 3e-4 1 0 || exit 1; fold eff 3e-4 0 0 || exit 1
A1=$(fauc eff_m1_lr3e-4_f0); A0=$(fauc eff_m0_lr3e-4_f0)
if better $A1 $A0 0.001; then UM=1; else UM=0; fi       # 元数据须至少 +0.001 才启用
echo "==> 元数据 A/B（第0折）：开=$A1  关=$A0  -> use_meta=$UM"

# ---------------- S2 ConvNeXt-S 学习率
fold cnxs 2e-4 $UM 0 || exit 1; fold cnxs 3e-4 $UM 0 || exit 1
C2=$(fauc cnxs_m${UM}_lr2e-4_f0); C3=$(fauc cnxs_m${UM}_lr3e-4_f0)
if better $C3 $C2 0; then CLR=3e-4; else CLR=2e-4; fi
echo "==> ConvNeXt-S 第0折：lr2e-4=$C2  lr3e-4=$C3  -> lr=$CLR"

# ---------------- S3 满 5 折
for k in $(seq 1 $((NF-1))); do fold eff 3e-4 $UM $k || exit 1; done
for k in $(seq 1 $((NF-1))); do fold cnxs $CLR $UM $k || exit 1; done

# ---------------- S4 标定 + 测速降级
G="--exp_dir $EXP --nfolds $NF --group eff=eff_m${UM}_lr3e-4 --group cnxs=cnxs_m${UM}_lr${CLR} --out_dir $ROOT/model_p3"
python calibrate_cv.py $G --tta 1 | tee $ROOT/logs/p3_calibrate.log; [ ${PIPESTATUS[0]} -eq 0 ] || exit 1

BENCH=$ROOT/bench_train
if [ ! -d $BENCH/p100 ]; then   # 速度基准用训练集图片（不用测试集做任何决定）
  python - <<EOF
import os, shutil, pandas as pd
df = pd.read_csv("$CACHE/train_index.csv").sample(105, random_state=0)
for i, r in enumerate(df.itertuples()):
    d = os.path.join("$BENCH", "p100" if i < 100 else "p5", r.Subject_id, r.Study_id)
    os.makedirs(d, exist_ok=True); shutil.copy(r.path, d)
EOF
fi
bench() { python $ROOT/cost.py none $BENCH/p100 $BENCH/p5 --model_dir $ROOT/model_p3 --no_check --json_out $ROOT/logs/p3_bench.json > $ROOT/logs/p3_bench.log 2>&1;
          python -c "import json;print(round(json.load(open('$ROOT/logs/p3_bench.json'))['avg_ms'],1))"; }
settta() { python -c "import json;p='$ROOT/model_p3/thresholds.json';d=json.load(open(p));d['tta']=bool($1);json.dump(d,open(p,'w'),indent=2,ensure_ascii=False)"; }
MS=$(bench); echo "==> 测速(训练集图片)：全部模型 + TTA = ${MS} ms"
if better $MS $LIMIT_MS 0.0001; then settta 0; MS=$(bench); echo "==> 关闭 TTA：${MS} ms"; fi
if better $MS $LIMIT_MS 0.0001; then
  python calibrate_cv.py $G --only eff --tta 1 | tee -a $ROOT/logs/p3_calibrate.log
  MS=$(bench); echo "==> 只用 EffV2-S 5折 + TTA：${MS} ms"
  if better $MS $LIMIT_MS 0.0001; then settta 0; MS=$(bench); echo "==> 只用 EffV2-S 5折、关 TTA：${MS} ms"; fi
fi

# ---------------- S5 推理 + 清单 + 核验
mkdir -p $ROOT/output_p3
python predict.py --data_root $DATA --model_dir $ROOT/model_p3 --out $ROOT/output_p3/submission.csv
python tools.py manifest --model_dir $ROOT/model_p3 --pretrain_dir $PRE
cd $ROOT
if [ -d pictures_100 ] && [ -d pictures_5 ]; then
  python cost.py output_p3/submission.csv pictures_100 pictures_5 --model_dir model_p3
else
  echo "[INFO] 未找到 pictures_100/pictures_5，耗时与一致性核验请手动运行 cost.py"
fi
echo "P3 完成 $(date '+%F %T')：output_p3/submission.csv"
