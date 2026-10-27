## 1. 任务目标
本次挑战赛聚焦于多标签胸部疾病分类任务。区别于传统的单标签分类，多标签分类要求算法能够同时识别一张胸部X光片中存在的多种异常征象或疾病类型。

参赛者需设计并训练深度学习模型，基于提供的标准化胸部X光片训练数据集，学习影像特征与对应多组标签间的复杂映射关系。最终，模型需要对测试集中的每张X光片进行精准推断，输出其可能包含的所有疾病类别概率，并确保在样本不均衡、标签共现及类间相似性高的挑战下，维持高鲁棒性与泛化能力。

## 2. 数据集描述

（1）数据构成与划分

●训练集（含标注）：共计18513次检查，用于模型训练，提供完整的多标签标注。

●验证集（含标注）：共计859次检查，用于调参及模型选型，提供完整的多标签标注。

●测试集（不含标注）：共计1414次检查，用于最终排名评估。测试集标签由主办方保密，参赛者需提交推理结果文件。

（2）数据内容与格式

●图⽚格式：JPG/PNG

●图⽚尺寸：无固定尺寸，因拍摄设备不同存在差异

●图片文件夹说明
images/
|-subject_00001/    # 单个患者的唯一id
| |-study_00001/    # 单个研究的唯一id
| | |-image_000001.jpg  # image_000001为图像id
| | |-image_000002.jpg
| |-study_00002/
| | |-image_000003.jpg
| | |-image_000004.jpg
....
### 类别标签
总共10个类别，每个类别只有在相关研究报告中被正面提及且出现在对应图像中，才会被标记。
1
纵隔心影增大
Enlarged Cardiomediastinum
2
气胸
Pneumothorax
3
肺部实变
Consolidation
4
肺炎
Pneumonia
5
肺水肿
Edema
6
心影扩大
Cardiomegaly
7
肺不张
Atelectasis
8
肺部阴影
Lung Opacity
9
胸腔积液
Pleural Effusion
10
无异常发现
No Finding


## 相关字段说明

●Subject_id：单个患者的唯一id

●Study_id：单个检查的唯一id，可能包含一张或多张胸部x光片

●Predict_class：患病类别id

●ViewPosition：X射线穿透身体的方向，即X光机的投射视角

●PatientOrientation：患者在检查时的身体姿势

●Rows：图像高度

●Columns：图像宽度

●Support Devices：是否存在任何类型的医疗支持设备，1表示有，0表示没有，-1表示不确定，None表示报告中未提及

## 评价标准
评估指标及排名方法

●主要指标：宏平均AUC (Macro-averaged AUC)，用于模型核心性能的衡量

●辅助指标：宏平均F1 (Macro-F1)，用于辅助区分性能相近的模型

●排名方法：优先以主要指标排名，当且仅当主要指标相等时，采用辅助指标进行二次排序



## 作品提交要求

1.提交内容

测试集最终预测结果的csv文件

说明：初赛仅提交预测结果csv文件，复赛需要同时提交预测结果csv文件及模型以供验证

2.提交文件格式

对于测试集中的每次检查即Study_id，模型需输出对应的类别id及其概率值。将最终结果生成⼀个.csv格式的结果⽂件，命名为"submission.csv"，编码为UTF-8，格式示例如下：

Subject_id  Study_id Predict_class Probability
100001       0000001       1          0.92
100001       0000001       2          0.87


说明：

●Study_id：单个检查的唯一id

●Predict_class：模型预测该次检查可能为某个类别id（默认以0.5为阈值），如果多个类别，可以输出多行

●Probability：模型预测该次检查为某类别的概率，取值范围为[0,1]

## 神经网络设计

## 3. Study-level 多输入轻量网络

代码严格以 `(Subject_id, Study_id)` 为一个样本，并且不使用任何预训练权重：

- 局部分支：原图使用 Lanczos 保细节缩放并填充至 `640×640`，按 2×2 网格切成四张 `320×320` 单通道图；共享轻量残差 CNN 为每块保留 `2×2` 空间特征，因此每张图得到 `4×4=16` 个局部 token；
- 全局分支：完整图像缩放并填充至 `320×320`，使用轻量残差网络得到全局特征；
- 元数据分支：恢复 v1 输入，编码 `ViewPosition`、`PatientOrientation`、原始高宽和长宽比；
- 融合：10 个可学习类别查询由全局特征和元数据条件化，对同一 Study 的全部局部 token 做交叉注意力，最后输出 10 个 logits。

四象限按图像坐标定义为左上、右上、左下、右下，只提供粗粒度位置先验；代码不会假定所有文件都能仅凭数组左右方向准确对应患者解剖左右侧。

同一 Study 的多张图不会被拆成独立训练样本：全局特征和元数据做均值汇聚，所有图片的四象限 token 拼接后供类别查询使用。CSV 使用标准库逐行解析，图片仅在 DataLoader 取样时按需读取。

开发电脑先运行极小规模的全链路检查（2 个训练 Study、2 个验证 Study、CPU、64 像素）：

```powershell
python testtrain.py
```

可先用流式 CSV 审计生成元数据报告；该命令不会解码完整图片集：

```powershell
python audit_data.py --data-root cxr_first --output data_audit.json
```

成功后会在 `outputs/dev_multiscale_smoke/年月日_时分秒/` 下保存本次运行。本机测试使用 `64×64` 全局图和 `128×128` 局部画布以减少 CPU 时间，网络结构和正式训练一致。`testtrain.py` 包含当前 Anaconda 环境所需的 OpenMP 兼容开关，只用于本机冒烟测试。

云端正式训练示例：

```bash
python train.py --data-root cxr_first --output-dir outputs/multiscale_v8 --epochs 50 --image-size 320 --local-canvas-size 640 --local-token-grid 2 --feature-dim 128 --dropout 0.1 --batch-size 64 --accum-studies 64 --progressive-augment --augmentation-ramp-start 3 --augmentation-ramp-end 15 --ranking-loss-weight 0.1 --ranking-loss-margin 0.2 --hard-negative-fraction 0.25 --ranking-ramp-start 5 --ranking-ramp-end 15 --workers 16 --device cuda --amp
```

建议在云端新建干净环境后安装 `requirements.txt`。模型从头训练，不依赖 torchvision，也不会下载任何外部权重。

训练目标为未加权 `BCEWithLogitsLoss + 类别级困难排序损失`。验证 loss 仍只计算 BCE，因而可与 v4.2 直接比较；训练日志分别记录 `train_bce_loss`、`train_ranking_loss` 和加权后的 `train_loss`。局部注意力输入与 v4.2 相同，每张图保留 16 个 token。

模型默认 `dropout=0.0`，交叉注意力、注意力残差和前馈网络中的 dropout 均关闭；如需重新启用，可显式传入例如 `--dropout 0.1`。

训练集恢复 v4.2 的固定组合渐进增强：第 1～3 轮为轻度增强，第 4～15 轮线性增加强度，第 15 轮后保持上限。完整强度为旋转 `±7°`、平移 `±4%`、缩放 `0.95～1.05`、亮度 `0.90～1.10`、对比度 `0.85～1.15` 和 Gamma `0.90～1.10`。验证集及训练集定期评估不做增强，日志中的 `augmentation_strength` 从 `0.0` 线性增加到 `1.0`。

困难排序以类别为单位，在每个 micro-batch 内把该类全部阳性与预测分数最高的 25% 阴性组成样本对，使用带 margin 的平滑排序损失；先对每类样本对取平均，再对有效类别做宏平均，避免高频类别主导。排序权重第 1～5 轮为 0，第 6～15 轮线性增加到 0.1。该损失要求 `--batch-size >= 2`，建议正式训练保持 64；梯度累积不会把不同 micro-batch 拼成排序样本对。`ranking_effective_classes` 和 `ranking_pairs_per_batch` 可用于检查每轮是否形成了足够多的有效排序对。

默认每隔 5 轮额外关闭数据增强并以 `eval()` 模式完整评估一次训练集，记录 `train_eval_loss`、`train_macro_auc`、`train_per_class_auc`、训练集 AP 和 F1；未到评估轮次时这些字段为 `null`。验证集字段仍为 `loss`、`macro_auc` 和 `per_class_auc`，`best.pt` 仍只按验证集 `macro_auc` 选择。可用 `--train-eval-interval 0` 关闭训练集评估。

每次新训练会在 `--output-dir` 下自动建立北京时间目录，例如：

```text
outputs/multiscale/20261024_153012/
├── config.json
├── metrics.jsonl
├── metrics.csv
├── best.pt
└── last.pt
```

`metrics.jsonl` 保存每轮完整指标，`metrics.csv` 将每类 AUC/AP 展开为独立列，便于使用 Excel 或绘图库查看。若验证指标尚不可计算，文件中记录为空值而不是无效的 `NaN` JSON。

中断后可在保持模型结构、图像尺寸及数据划分不变的前提下恢复；`--epochs` 表示最终总轮数：

```bash
python train.py --data-root cxr_first --epochs 50 --image-size 320 --local-canvas-size 640 --device cuda --amp --resume outputs/multiscale/20261024_153012/last.pt
```

测试集推理同时保留 dense 与阈值版结果：

```bash
python predict.py --checkpoint outputs/multiscale/20261024_153012/best.pt --data-root cxr_first --device cuda --amp
```

输出为 `outputs/predictions/submission_dense.csv` 和 `submission_thresholded.csv`。由于官方说明中 Macro-AUC 与“仅提交超过 0.5 的类别”存在歧义，提交前需根据官方样例或评测答疑确定最终采用哪一种。


## process
v1.0 20261024_222916   epoch22 0.769
v2.0 增加2*2网格局部输入精度320->512 
v3.0 回退到1.0精度处理不变，去掉结构化数据图片长宽和长宽比，启用裁剪后的 sqrt_ratio pos_weight
v4.0 恢复 v1 元数据与未加权 BCE；每个局部块保留 2×2 特征，四块共 16 个局部 token         epoch26 0.752
python train.py --data-root cxr_first --output-dir outputs/multiscale \
  --epochs 50 --image-size 320 --local-canvas-size 640 \
  --local-token-grid 2 --feature-dim 128 \
  --batch-size 64 --accum-studies 64 \
  --train-eval-interval 5 \
  --workers 12 --device cuda --amp
v4.1 增加--dropout 0.1   epoch21 0.7684
v4.2 在 v4.1 基础上增加第 3～15 轮渐进增强 epoch27 0.771 "time": "2026-10-26T15:40:24.095566+08:00"

v5.0 在 v4.2 基础上保留每图 16 个局部 token，并为每张图增加 1 个带
`ViewPosition` 与 `PatientOrientation` embedding 的全局 token。每图共 17 个 token
一起进入类别查询交叉注意力，使不同疾病能够分别选择 AP、PA、LATERAL 或 LL 视图。
默认仍保留原有 Study 全局特征与元数据均值作为 residual；使用
`--no-study-mean-residual` 可做完全移除均值汇聚的消融实验，使用
`--no-global-tokens` 可回到旧版仅局部 token 的注意力结构。

v5.0 正式训练示例：

```bash
python train.py --data-root cxr_first --output-dir outputs/multiscale_v5 \
  --epochs 50 --image-size 320 --local-canvas-size 640 \
  --local-token-grid 2 --feature-dim 128 --dropout 0.1 \
  --global-tokens --study-mean-residual \
  --batch-size 64 --accum-studies 64 \
  --progressive-augment --augmentation-ramp-start 3 --augmentation-ramp-end 15 \
  --train-eval-interval 5 --workers 12 --device cuda --amp
```

推理时可导出每个类别在每张图上的全局、局部和总注意力。权重已对 attention head
取平均，同一 Study、同一类别下所有图像的 `Total_attention` 之和为 1：

```bash
python predict.py --checkpoint outputs/multiscale_v5/运行目录/best.pt \
  --data-root cxr_first --device cuda --amp \
  --attention-output outputs/predictions/v5_attention_views.csv
```
v6.0 回退到 v4.2 网络结构：移除 v5.0 的逐图 global token，恢复每图 16 个局部
token 进入类别查询注意力，以及全局特征与元数据的 Study 均值 residual。保留
`--dropout 0.1` 和第 3～15 轮渐进增强，并默认启用标签数量采样课程：

| 阶段 | No Finding | 单标签 | 双标签 | ≥3 标签 |
| --- | ---: | ---: | ---: | ---: |
| epoch 1～3 | 30% | 50% | 15% | 5% |
| epoch 4～15 | 线性过渡 | 线性下降 | 线性增加 | 线性增加 |
| epoch 16 以后 | 36.1% | 26.5% | 19.8% | 17.6% |

epoch 1～15 每轮保持与训练集相同的 Study 数，按目标比例进行分层抽样；epoch 16
以后恢复全训练集无放回随机遍历。验证集和定期 train-eval 始终使用原始分布。
`metrics.jsonl` 与 `metrics.csv` 会记录每组目标概率和实际抽样数量。可使用
`--no-label-cardinality-curriculum` 做自然采样对照。

```bash
python train.py --data-root cxr_first --output-dir outputs/multiscale_v6 \
  --epochs 50 --image-size 320 --local-canvas-size 640 \
  --local-token-grid 2 --feature-dim 128 --dropout 0.1 \
  --label-cardinality-curriculum \
  --label-curriculum-ramp-start 3 --label-curriculum-ramp-end 15 \
  --batch-size 64 --accum-studies 64 \
  --progressive-augment --augmentation-ramp-start 3 --augmentation-ramp-end 15 \
  --train-eval-interval 5 --workers 12 --device cuda --amp
```

v7.0 再次回退到 v4.2 的网络、自然采样和未加权 BCE，仅将固定组合渐进增强替换为
动态成像增强。完整强度下保留 15% 原图，仿射以 70% 概率执行，曝光增强每次随机
选择 1～2 项；分辨率退化、噪声、模糊和锐化的概率分别渐进至 20%、25%、10%
和 10%。模糊与锐化互斥，且不使用水平翻转、随机擦除或 CutMix。

```bash
python train.py --data-root cxr_first --output-dir outputs/multiscale_v7 \
  --epochs 50 --image-size 320 --local-canvas-size 640 \
  --local-token-grid 2 --feature-dim 128 --dropout 0.1 \
  --batch-size 64 --accum-studies 64 \
  --progressive-augment --dynamic-imaging-augment \
  --augmentation-ramp-start 3 --augmentation-ramp-end 15 \
  --train-eval-interval 5 --workers 12 --device cuda --amp
```

v8.0 回退到 v4.2 的网络结构、自然采样和固定组合渐进增强，并将训练目标改为
`BCE + 类别级困难排序损失`。对每个类别选取当前得分最高的 25% 阴性作为 hard
negatives，与 batch 内该类全部阳性比较；类别内平均后再对有效类别宏平均。排序权重
在第 6～15 轮由 0 线性增加到 0.1，避免训练早期不稳定。验证 loss 保持纯 BCE。

从头训练（推荐作为主对照）：

```bash
python train.py --data-root cxr_first --output-dir outputs/multiscale_v8 \
  --epochs 50 --image-size 320 --local-canvas-size 640 \
  --local-token-grid 2 --feature-dim 128 --dropout 0.1 \
  --batch-size 64 --accum-studies 64 \
  --progressive-augment --augmentation-ramp-start 3 --augmentation-ramp-end 15 \
  --ranking-loss-weight 0.1 --ranking-loss-margin 0.2 \
  --hard-negative-fraction 0.25 --ranking-ramp-start 5 --ranking-ramp-end 15 \
  --train-eval-interval 5 --workers 12 --device cuda --amp
```

可用 v4.2 的 `best.pt` 做同结构热启动。`--init-checkpoint` 只载入模型权重，并重置
优化器、学习率调度器和 epoch；它与继续旧训练状态的 `--resume` 不同且不能同时使用。
建议把热启动作为第二组实验，降低学习率并缩短训练，而不是再以 `3e-4` 完整训练 50 轮：

```bash
python train.py --data-root cxr_first --output-dir outputs/multiscale_v8_from_v42 \
  --epochs 10 --image-size 320 --local-canvas-size 640 \
  --local-token-grid 2 --feature-dim 128 --dropout 0.1 \
  --batch-size 64 --accum-studies 64 --lr 3e-5 --warmup-epochs 0.5 \
  --progressive-augment --augmentation-ramp-start 3 --augmentation-ramp-end 15 \
  --ranking-loss-weight 0.1 --ranking-loss-margin 0.2 \
  --hard-negative-fraction 0.25 --ranking-ramp-start 0 --ranking-ramp-end 2 \
  --init-checkpoint outputs/multiscale_v42/best.pt \
  --train-eval-interval 5 --workers 12 --device cuda --amp
```

热启动要求类别顺序、图像尺寸、局部 token 网格、特征维度、dropout 和预处理配置
与 v4.2 checkpoint 完全一致，代码会在不兼容时直接报错。由于该权重已见过同一训练集，
严格说它是 fine-tuning/warm start，而不是独立数据预训练；应与从头训练使用相同划分和
随机种子比较，并以验证集 Macro-AUC 判断是否真正改善泛化。
