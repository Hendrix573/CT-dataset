#!/usr/bin/env bash
# P1 自动调参（断点续跑：已完成的实验自动跳过）
#  R1 EffV2-S: epochs x lr
#  R2 EffV2-S: drop_path x weight_decay
#  R3 EffV2-S: BCE / 关闭翻转 / 多训2轮
#  R4 ConvNeXt-S、ConvNeXt-T 继承最优配置，各试 2 个 lr
#  R5 各骨干最优 -> model_p1/ -> 集成子集选择 -> 阈值标定 -> 预测 -> 清单 -> 耗时
# 产物在 exp/ model_p1/ output_p1/，不覆盖现有 model/ 和 submission.csv
set -uo pipefail
cd "$(dirname "$0")/code"
mkdir -p ../exp ../logs
CACHE=../cache/576
PRE=../pretrain_model
DATA=../../cxr_first
WORKERS=${WORKERS:-8}
declare -A BB=( [eff]=tf_efficientnetv2_s.in21k_ft_in1k [cnxs]=convnext_small.fb_in22k_ft_in1k [cnxt]=convnext_tiny.fb_in22k_ft_in1k )
declare -A BS=( [eff]=24 [cnxs]=24 [cnxt]=32 )

run() {  # run <arch> <epochs> <lr> <drop_path> <wd> <loss> <hflip>
  local a=$1 ep=$2 lr=$3 dp=$4 wd=$5 loss=$6 hf=$7
  local tag=${a}_e${ep}_lr${lr}_dp${dp}_wd${wd}_${loss}_hf${hf}
  local out=../exp/$tag
  if [ -f "$out/DONE" ]; then echo "[skip] $tag"; return; fi
  mkdir -p "$out"
  echo "$a $ep $lr $dp $wd $loss $hf" > "$out/params.txt"
  echo "[run ] $tag   $(date '+%F %T')"
  if python train.py --backbone ${BB[$a]} --pretrained_path $PRE/${BB[$a]}.safetensors \
       --cache_dir $CACHE --img_size 512 --bs ${BS[$a]} --workers $WORKERS --seed 7 \
       --epochs $ep --lr $lr --drop_path $dp --wd $wd --loss $loss --hflip $hf \
       --out_dir "$out" > ../logs/$tag.log 2>&1; then
    touch "$out/DONE"; python ../p1_tools.py show $tag
  else
    echo "[FAIL] $tag，见 logs/$tag.log"
  fi
}
best() { python ../p1_tools.py best "$1"; }

# ---------------- R1：epochs x lr（drop_path 0.3, wd 0.05, ASL, 翻转 0.5）
for EP in 8 10 12; do for LR in 1e-4 1.5e-4 3e-4; do
  run eff $EP $LR 0.3 0.05 asl 0.5
done; done
read A EP LR DP WD LOSS HF TAG AUC BEP <<< "$(best eff)"
echo "==> R1 最优: $TAG  AUC=$AUC  best_ep=$BEP"

# ---------------- R2：drop_path x wd
for D in 0.2 0.3 0.4; do for W in 0.05 0.1; do
  run eff $EP $LR $D $W asl 0.5
done; done
read A EP LR DP WD LOSS HF TAG AUC BEP <<< "$(best eff)"
echo "==> R2 最优: $TAG  AUC=$AUC  best_ep=$BEP"

# ---------------- R3：损失 / 翻转 / 训练长度
run eff $EP $LR $DP $WD bce 0.5
run eff $EP $LR $DP $WD asl 0.0
run eff $((EP + 2)) $LR $DP $WD asl 0.5
read A EP LR DP WD LOSS HF TAG AUC BEP <<< "$(best eff)"
echo "==> EffV2-S 最终最优: $TAG  AUC=$AUC  best_ep=$BEP"

# ---------------- R4：ConvNeXt 继承 epochs / drop_path / wd / loss / hflip
for L in 5e-5 1e-4;     do run cnxs $EP $L $DP $WD $LOSS $HF; done
for L in 7.5e-5 1.5e-4; do run cnxt $EP $L $DP $WD $LOSS $HF; done

echo; echo "================ 全部实验排行 ================"
python ../p1_tools.py table

# ---------------- R5：组装 model_p1 -> 子集选择 -> 标定 -> 预测
rm -rf ../model_p1 && mkdir -p ../model_p1
for a in eff cnxs cnxt; do
  LINE=$(best $a) || continue
  read _ _ _ _ _ _ _ T _ _ <<< "$LINE"
  mkdir -p ../model_p1/$a
  cp ../exp/$T/{best.pth,valid_study_preds.csv,valid_study_labels.npy,train_log.json,params.txt} ../model_p1/$a/
  echo "$a <- $T"
done
echo "---- 集成子集（验证集 macro-AUC）----"
python ../p1_tools.py subset ../model_p1 | tee ../logs/p1_subset.log
KEEP=$(tail -1 ../logs/p1_subset.log | sed 's/^KEEP //')
for d in ../model_p1/*/; do
  n=$(basename "$d")
  [[ " $KEEP " == *" $n "* ]] || { echo "移出集成: $n"; mv "$d" ../exp/_dropped_$n; }
done

python calibrate.py --model_dir ../model_p1 --tta 1
mkdir -p ../output_p1
python predict.py --data_root $DATA --model_dir ../model_p1 --out ../output_p1/submission.csv
python tools.py manifest --model_dir ../model_p1 --pretrain_dir $PRE

cd ..
if [ -d pictures_100 ] && [ -d pictures_5 ]; then
  python cost.py output_p1/submission.csv pictures_100 pictures_5 --model_dir model_p1
else
  echo "[INFO] 未找到 pictures_100/pictures_5，耗时测试请手动运行 cost.py"
fi
echo "P1 完成 $(date '+%F %T')：output_p1/submission.csv"
