# -*- coding: utf-8 -*-
"""模型定义 + 推理器。predict.py 与 cost.py 共用 Predictor，保证推理逻辑完全一致。
P2：可选视角元数据分支（ViewPosition/PatientOrientation one-hot -> MLP -> 与图像特征拼接）。
兼容旧 checkpoint（无元数据分支）。"""
import os
import glob

import numpy as np
import torch
import torch.nn as nn
import timm

from utils import (NUM_CLASSES, META_DIM, load_gray, to_model_input, aggregate_study,
                   remap_probs, load_json, meta_vector)


def load_local_pretrained(backbone, path):
    """离线加载 ImageNet 权重（.safetensors / .pth）：RGB 首层卷积合并为单通道，丢弃分类头，绝不联网。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"预训练权重不存在: {path}")
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(path)
    else:
        sd = torch.load(path, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd.get("model", sd))
    from timm.models import adapt_input_conv
    own = backbone.state_dict()
    out, skipped = {}, []
    for k, v in sd.items():
        if k not in own:
            skipped.append(k)
            continue
        if v.ndim == 4 and own[k].ndim == 4 and v.shape[1] == 3 and own[k].shape[1] == 1:
            v = adapt_input_conv(1, v)
        if v.shape != own[k].shape:
            skipped.append(k)
            continue
        out[k] = v
    missing = [k for k in own if k not in out]
    backbone.load_state_dict(out, strict=False)
    print(f"[pretrain] loading local pretrained weights: {path} -> {len(out)}/{len(own)} 参数，"
          f"缺失 {len(missing)}，忽略 {len(skipped)}", flush=True)
    if len(out) < 0.9 * len(own):
        raise RuntimeError(f"权重与骨干不匹配，缺失示例: {missing[:5]}，请确认 --backbone 与文件对应")


class CXRNet(nn.Module):
    """timm 骨干（ImageNet 预训练，单通道）+ GeM 池化 + [可选元数据分支] + 多标签线性头。"""

    def __init__(self, backbone, pretrained=False, pretrained_path=None, drop_rate=0.2,
                 drop_path_rate=0.1, use_meta=False, meta_hidden=64):
        super().__init__()
        kw = dict(pretrained=pretrained and not pretrained_path, in_chans=1, num_classes=0,
                  global_pool="", drop_path_rate=drop_path_rate)
        try:
            self.backbone = timm.create_model(backbone, **kw)
        except TypeError:
            kw.pop("drop_path_rate")
            self.backbone = timm.create_model(backbone, **kw)
        if pretrained_path:
            load_local_pretrained(self.backbone, pretrained_path)
        self.nf = self.backbone.num_features
        self.use_meta = use_meta
        self.p = nn.Parameter(torch.ones(1) * 3.0)
        self.drop = nn.Dropout(drop_rate)
        if use_meta:
            self.meta_fc = nn.Sequential(nn.Linear(META_DIM, meta_hidden), nn.GELU())
            self.head = nn.Linear(self.nf + meta_hidden, NUM_CLASSES)
        else:
            self.head = nn.Linear(self.nf, NUM_CLASSES)

    def pool(self, x, eps=1e-6):
        if x.ndim == 3:
            return x.mean(1)
        if x.shape[1] != self.nf and x.shape[-1] == self.nf:
            x = x.permute(0, 3, 1, 2)
        x = x.float().clamp(min=eps).pow(self.p).mean((-2, -1)).pow(1.0 / self.p)
        return x.to(self.head.weight.dtype)

    def forward(self, x, meta=None):
        f = self.drop(self.pool(self.backbone.forward_features(x)))
        if self.use_meta:
            if meta is None:
                meta = torch.from_numpy(meta_vector(None, None)).to(f.device).expand(f.shape[0], -1)
            f = torch.cat([f, self.meta_fc(meta.to(f.dtype))], 1)
        return self.head(f)


def build_from_cfg(cfg, **kw):
    return CXRNet(cfg["backbone"], use_meta=cfg.get("use_meta", False), **kw)


def save_checkpoint(model, cfg, out_dir, tag="best"):
    """推理权重以 fp16 保存（体积减半，计入 500MB 限额）。"""
    os.makedirs(out_dir, exist_ok=True)
    sd = {k: (v.half() if v.is_floating_point() else v) for k, v in model.state_dict().items()}
    torch.save({"state_dict": sd, "cfg": cfg}, os.path.join(out_dir, f"{tag}.pth"))


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    m = build_from_cfg(cfg, pretrained=False, drop_rate=0.0, drop_path_rate=0.0)
    m.load_state_dict({k: v.float() for k, v in ck["state_dict"].items()})
    m.eval().to(device)
    return m, cfg


class Predictor:
    """加载 model_dir 下所有 */best.pth 组成集成（含 5 折模型）；读取 thresholds.json：
    thresholds / agg / frontal_w / tta。eval 模式、参数冻结、无随机增强；TTA 为确定性水平翻转。"""

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
        self.frontal_w = float(meta.get("frontal_w", 1.0))
        self.tta = meta.get("tta", True) if tta is None else tta
        self.cache_size = self.cfgs[0]["cache_size"]
        assert all(c["cache_size"] == self.cache_size for c in self.cfgs), "集成模型须使用同一 cache_size"
        self.uses_meta = any(c.get("use_meta", False) for c in self.cfgs)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.precision = "fp16" if self.fp16 else "fp32"

    @torch.inference_mode()
    def predict_image(self, path, view=None, orient=None):
        """单张图 -> 10 类原始概率（集成 + TTA 平均，未重映射），batch=1。
        view/orient 为该图自身的元数据（来自官方 csv），仅元数据模型使用。"""
        gray = load_gray(path, self.cache_size)
        mv = torch.from_numpy(meta_vector(view, orient))[None].to(self.device)
        cache = {}
        probs = []
        for m, cfg in zip(self.models, self.cfgs):
            s = cfg["img_size"]
            if s not in cache:
                x = torch.from_numpy(to_model_input(gray, s))[None].to(self.device)
                if self.fp16:
                    x = x.half()
                if self.tta:
                    x = torch.cat([x, torch.flip(x, dims=[3])], 0)
                cache[s] = x
            x = cache[s]
            mm = mv.expand(x.shape[0], -1) if m.use_meta else None
            probs.append(torch.sigmoid(m(x, mm).float()).mean(0))
        return torch.stack(probs).mean(0).cpu().numpy().astype(np.float64)

    def finalize(self, p):
        return remap_probs(p, self.thresholds) if self.thresholds is not None else np.asarray(p)

    def finalize_study(self, image_probs, views=None):
        """检查内多图概率 -> 检查级最终概率（确定性聚合 + 阈值重映射）"""
        return self.finalize(aggregate_study(np.stack(image_probs), self.agg, views, self.frontal_w))
