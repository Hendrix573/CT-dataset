# 多标签胸片疾病分类 — 参赛方案

## 1. 方案结论（为什么这样做最容易拿高分）

| 决策 | 选择 | 理由 |
|---|---|---|
| 骨干 | ConvNeXt-S（IN22k→1k）+ EfficientNetV2-S（IN21k→1k）+ ConvNeXt-T | 非医学、Apache-2.0/MIT、非 gated。CheXpert 类任务上 CNN 在中等数据量下比 ViT 更稳。三个不同结构集成，通常比单模型 macro-AUC 高 0.5~1.5 个点 |
| 分辨率 | 512（缓存 576，随机裁剪） | 气胸、小片实变需要细节；512 是精度与 100ms 限时的平衡点 |
| 输入 | 单通道灰度 | timm 自动把 RGB 首层卷积合并为单通道，减少 1/3 首层计算 |
| 损失 | ASL（γ-=2，clip 0.05） | 10 类强不平衡（No Finding/Lung Opacity 多，Pneumothorax/Enlarged CM 少） |
| 训练 | AdamW + 余弦 + warmup，AMP，EMA，GeM 池化，头部 5× LR | 都是稳定涨点、零推理开销的技巧 |
| 增强 | 旋转±10°、缩放、平移、裁剪、亮度/对比度/gamma、CoarseDropout、水平翻转 | 规则允许；推理时全部关闭 |
| 检查级聚合 | 0.5×mean + 0.5×max（同一 Study 多张图） | 标签按检查给出；侧位片能补充正位看不清的积液/心影。确定性聚合，规则允许 |
| TTA | 水平翻转平均 | 规则允许，+0.2~0.5 AUC |
| 副指标 F1 | 验证集逐类找 F1 最优阈值 t，用单调分段线性映射把 t 对齐到 0.5 | 单调变换不改变 AUC，但主办方按 0.5 判阳性时 F1 达到验证集最优。平局时靠它胜出 |
| 验证集 | 先用 valid 选 epoch、标定阈值，再（可选）并入训练 | 符合第 3 条：所有选择都在并入前完成 |
| 推理速度 | JPEG DCT 域降采样解码（PIL draft）+ fp16 + batch=1 | 原图约 3000px，普通解码约 40~80ms，是最大瓶颈；draft 解码可降到约 5~10ms |
| 体积 | fp16 存权重：约 100+43+56 ≈ 200MB | 低于 500MB 上限 |

没有采用的方案：
- 医学预训练权重（如 CheXpert/MIMIC 预训练、RETFound、BiomedCLIP）：违规。
- 在测试集上做伪标签、TTA 统计或阈值调整：违规。
- 官方训练集自监督预训练（规则允许）：收益通常 <0.5 点，GPU 成本高，作为后续可选项。

## 2. 目录

```
cxr_competition/
├── code/
│   ├── utils.py        # 标签解析、确定性读图、AUC/F1、阈值重映射、检查级聚合
│   ├── model.py        # CXRNet + Predictor（predict.py 与 cost.py 共用）
│   ├── preprocess.py   # 离线缓存 train/valid
│   ├── train.py        # 训练（阶段1 / 阶段2 --merge_valid）
│   ├── calibrate.py    # 集成 + 阈值标定（仅 valid）
│   ├── predict.py      # 生成 submission.csv + image_preds.csv
│   ├── tools.py        # 下载预训练权重 / 文件清单 + SHA-256 + 500MB 自检
│   └── requirements.txt
├── model/              # 推理权重 */best.pth（fp16）+ thresholds.json
├── pretrain_model/     # ImageNet 预训练权重 + 来源/许可证 json
├── licenses/           # 许可证全文（Apache-2.0 / MIT）
├── cost.py             # 耗时与一致性核验
├── run_all.sh          # 一键训练与推理
└── smoke_test.py       # 合成数据全流程冒烟测试
```

## 3. 运行

数据放在项目同级目录：`cxr_competition/../cxr_first/{train,valid,test}`，以及 `train.csv`、`valid.csv`、`test.csv`。

```bash
pip install -r code/requirements.txt
bash run_all.sh                         # 下载权重 → 缓存 → 训练3模型 → 标定 → 预测 → 清单
python cost.py submission.csv pictures_100 pictures_5
python smoke_test.py                    # 无数据/无 GPU 也能跑的流程自检
```

参考耗时（单卡 4090）：缓存约 20 分钟；每个模型 20 epoch 约 2~3 小时。

## 4. 调参优先级（按 valid macro-AUC 判断）

1. 分辨率 512 → 640（cache_size 改为 704）。小病灶类通常涨点，但要复测 cost.py 是否仍 ≤100ms。
2. epochs 15/20/30，lr ×0.5 或 ×2。
3. `--loss bce` 与 `asl` 对比。
4. `--hflip 0` 与 `0.5` 对比。心脏位置左右不对称，有时关闭翻转更好。
5. 同一结构换不同 seed 再加一个集成成员。

## 5. 推理限时

每张图的耗时 ≈ 解码（约 5~10ms）+ Σ模型前向 ×（TTA 时 ×2）。
- 如果 cost.py 超过 100ms：先在 `model/thresholds.json` 里把 `"tta"` 设为 false，或去掉最慢的集成成员。
- 改完后必须重新运行 predict.py，保证 submission.csv 与 cost.py 输出一致。

## 6. 合规材料

- `python code/tools.py manifest` 会生成 `file_manifest.json`，包含每个文件的 SHA-256 和大小。
- `pretrain_model/*.json` 记录权重来源链接（HF Hub）、许可证和预训练数据集。
- 许可证全文放到 `licenses/`：timm 权重用 Apache-2.0，ConvNeXt 原始权重用 MIT。
- 推理精度：fp16（GPU）。
- 验证集使用说明：阶段1只用于选择 epoch 和阈值；如果做了阶段2，要申报“验证集已并入训练，epoch 和阈值在并入前确定”。
- 后处理说明：同一 Study 内多张图做确定性 mean/max 聚合，再做固定的单调阈值映射。没有用到任何测试集统计量。
