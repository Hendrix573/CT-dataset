#!/usr/bin/env bash
# 完整训练 + 推理流水线（单卡 24GB 级 GPU，如 RTX 3090/4090/A10/A100）。
# 数据位置：项目同级目录 ../cxr_first（即 cxr_competition/../cxr_first）
set -euo pipefail
cd "$(dirname "$0")/code"
DATA=../../cxr_first
CACHE=../cache
SIZE=576          # 缓存边长；训练随机裁剪/缩放到 512
WORKERS=${WORKERS:-8}

# 0) 预训练权重：优先使用 pretrain_model/<backbone>.safetensors（可在其他电脑下载后拷贝过来）
#    缺失时才尝试联网下载；随后记录 SHA-256 / 来源 / 许可证
PRE=../pretrain_model
mkdir -p $PRE
for BB in convnext_small.fb_in22k_ft_in1k tf_efficientnetv2_s.in21k_ft_in1k convnext_tiny.fb_in22k_ft_in1k; do
  F=$PRE/$BB.safetensors
  if [ ! -f "$F" ]; then
    echo "[INFO] $F 不存在，尝试下载（可设置 HF_ENDPOINT=https://hf-mirror.com）"
    curl -L --fail -o "$F" "${HF_ENDPOINT:-https://huggingface.co}/timm/$BB/resolve/main/model.safetensors"
  fi
  python tools.py register --backbone $BB --file "$F"
done

# 1) 离线缓存（仅 train/valid）
python preprocess.py --data_root $DATA --cache_dir $CACHE --cache_size $SIZE --workers $WORKERS

# 2) 阶段1：train 训练，valid 选最优 epoch，产出 valid 预测
python train.py --cache_dir $CACHE/$SIZE --out_dir ../model/cnx_s_512 \
  --backbone convnext_small.fb_in22k_ft_in1k --pretrained_path ../pretrain_model/convnext_small.fb_in22k_ft_in1k.safetensors \
  --img_size 512 --epochs 20 --bs 24 --lr 1e-4 --workers $WORKERS --seed 42
python train.py --cache_dir $CACHE/$SIZE --out_dir ../model/effv2s_512 \
  --backbone tf_efficientnetv2_s.in21k_ft_in1k --pretrained_path ../pretrain_model/tf_efficientnetv2_s.in21k_ft_in1k.safetensors \
  --img_size 512 --epochs 20 --bs 24 --lr 3e-4 --workers $WORKERS --seed 7
python train.py --cache_dir $CACHE/$SIZE --out_dir ../model/cnx_t_512 \
  --backbone convnext_tiny.fb_in22k_ft_in1k --pretrained_path ../pretrain_model/convnext_tiny.fb_in22k_ft_in1k.safetensors \
  --img_size 512 --epochs 20 --bs 32 --lr 1.5e-4 --workers $WORKERS --seed 2024

# 3) 用 valid 标定集成阈值（必须在任何“合并 valid 训练”之前完成）
python calibrate.py --model_dir ../model --tta 1

# 4)（可选）阶段2：train+valid 按阶段1最优 epoch 重训，覆盖 best.pth；thresholds.json 保持阶段1结果
#   EP=$(python -c "import torch;print(torch.load('../model/cnx_s_512/best.pth',weights_only=False)['cfg']['best_epoch'])")
#   python train.py ... --merge_valid --epochs $EP

# 5) 测试集推理 -> ../submission.csv + ../image_preds.csv
python predict.py --data_root $DATA --model_dir ../model --out ../submission.csv

# 6) 文件清单 / SHA-256 / 500MB 自检
python tools.py manifest --model_dir ../model --pretrain_dir ../pretrain_model

# 7) 耗时与一致性核验（主办方命令格式）
# cd .. && python cost.py submission.csv pictures_100 pictures_5
