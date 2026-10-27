# -*- coding: utf-8 -*-
"""模型定义 + 推理器。predict.py 与 cost.py 共用 Predictor，保证推理逻辑完全一致。"""
import os
import glob

import numpy as np
import torch
import torch.nn as nn
import timm

from utils import (NUM_CLASSES, load_gray, to_model_input, aggregate_study,
                   remap_probs, load_json)


class CXRNet(nn.Module):
    """timm 骨干（ImageNet 预训练，单通道输入）+ GeM 池化 + 多标签线性头。"""

    def __init__(self, backbone, pretrained=False, pretrained_path=None, drop_rate=0.2, drop_path_rate=0.1):
        super().__init__()
        kw = dict(pretrained=pretrained, in_chans=1, num_classes=0, global_pool="",
                  drop_path_rate=drop_path_rate)
        if pretrained_path:  # 离线权重（提交包 pretrain_model/ 下）；timm 自动把 RGB 首层卷积合并为单通道
            kw["pretrained"] = True
            kw["pretrained_cfg_overlay"] = dict(file=pretrained_path)
        try:
            self.backbone = timm.create_model(backbone, **kw)
        except TypeError:  # 部分模型不支持 drop_path_rate
            kw.pop("drop_path_rate")
            self.backbone = timm.create_model(backbone, **kw)
        self.nf = self.backbone.num_features
        self.p = nn.Parameter(torch.ones(1) * 3.0)
        self.drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.nf, NUM_CLASSES)

    def pool(self, x, eps=1e-6):
        if x.ndim == 3:  # ViT tokens (B,N,C)
            return x.mean(1)
        if x.shape[1] != self.nf and x.shape[-1] == self.nf:
            x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        x = x.float().clamp(min=eps).pow(self.p).mean((-2, -1)).pow(1.0 / self.p)
        return x.to(self.head.weight.dtype)

    def forward(self, x):
        return self.head(self.drop(self.pool(self.backbone.forward_features(x))))


def save_checkpoint(model, cfg, out_dir, tag="best"):
    """推理权重以 fp16 保存（体积减半，计入 500MB 限额）。"""
    os.makedirs(out_dir, exist_ok=True)
    sd = {k: (v.half() if v.is_floating_point() else v) for k, v in model.state_dict().items()}
    torch.save({"state_dict": sd, "cfg": cfg}, os.path.join(out_dir, f"{tag}.pth"))


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    m = CXRNet(cfg["backbone"], pretrained=False, drop_rate=0.0, drop_path_rate=0.0)
    m.load_state_dict({k: v.float() for k, v in ck["state_dict"].items()})
    m.eval().to(device)
    return m, cfg


class Predictor:
    """加载 model_dir 下所有 */best.pth 组成集成；读取 thresholds.json 做单调概率重映射。
    eval 模式、参数冻结、无随机增强；TTA 为确定性水平翻转（规则第6条允许）。"""

    def __init__(self, model_dir, device=None, fp16=True, tta=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.fp16 = bool(fp16) and self.device.type == "cuda"
        ckpts = sorted(glob.glob(os.path.join(model_dir, "*", "best.pth")))
        if not ckpts:
            raise FileNotFoundError(f"{model_dir} 下没有 */best.pth")
        self.models, self.cfgs = [], []
        for c in ckpts:
            m, cfg = load_model(c, self.device)
            if self.fp16:
                m.half()
            self.models.append(m)
            self.cfgs.append(cfg)
        self.ckpts = ckpts
        thr_path = os.path.join(model_dir, "thresholds.json")
        meta = load_json(thr_path) if os.path.exists(thr_path) else {}
        self.thresholds = meta.get("thresholds")
        self.agg = meta.get("agg", "meanmax")
        self.tta = meta.get("tta", True) if tta is None else tta
        self.cache_size = self.cfgs[0]["cache_size"]
        assert all(c["cache_size"] == self.cache_size for c in self.cfgs), "集成模型须使用同一 cache_size"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.precision = "fp16" if self.fp16 else "fp32"

    @torch.inference_mode()
    def predict_image(self, path):
        """单张图 -> 10 类原始概率（集成+TTA 平均，未重映射），batch=1"""
        gray = load_gray(path, self.cache_size)
        probs = []
        for m, cfg in zip(self.models, self.cfgs):
            x = torch.from_numpy(to_model_input(gray, cfg["img_size"]))[None].to(self.device)
            if self.fp16:
                x = x.half()
            if self.tta:
                x = torch.cat([x, torch.flip(x, dims=[3])], 0)
            probs.append(torch.sigmoid(m(x).float()).mean(0))
        return torch.stack(probs).mean(0).cpu().numpy().astype(np.float64)

    def finalize(self, p):
        return remap_probs(p, self.thresholds) if self.thresholds is not None else np.asarray(p)

    def finalize_study(self, image_probs):
        """检查内多图概率 -> 检查级最终概率（确定性聚合 + 阈值重映射）"""
        return self.finalize(aggregate_study(np.stack(image_probs), self.agg))
