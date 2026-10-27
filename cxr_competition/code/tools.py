# -*- coding: utf-8 -*-
"""辅助工具：
  python tools.py download --backbone convnext_small.fb_in22k_ft_in1k --out ../pretrain_model
      离线保存 ImageNet 预训练权重 + 记录 HF 来源链接与许可证（提交材料用）
  python tools.py manifest --model_dir ../model --pretrain_dir ../pretrain_model
      生成文件清单、SHA-256、推理权重总大小（规则第8条 ≤500MB 自检）"""
import os
import sys
import glob
import hashlib
import argparse

from utils import save_json

# 推荐骨干（均为通用自然图像预训练、非 NC、非 gated、可再分发）
LICENSES = {
    "convnext": ("Apache-2.0 (timm 权重) / MIT (facebookresearch/ConvNeXt)", "ImageNet-22k -> ImageNet-1k"),
    "tf_efficientnetv2": ("Apache-2.0", "ImageNet-21k -> ImageNet-1k"),
    "efficientnet": ("Apache-2.0", "ImageNet-1k"),
    "resnet": ("Apache-2.0", "ImageNet-1k"),
}


def sha256(path, bs=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            h.update(b)
    return h.hexdigest()


def download(args):
    import timm
    import torch
    os.makedirs(args.out, exist_ok=True)
    m = timm.create_model(args.backbone, pretrained=True)
    cfg = m.pretrained_cfg
    path = os.path.join(args.out, args.backbone.replace("/", "_") + ".pth")
    torch.save(m.state_dict(), path)
    fam = next((k for k in LICENSES if args.backbone.startswith(k)), None)
    info = dict(backbone=args.backbone, file=os.path.basename(path), sha256=sha256(path),
                hf_hub_id=cfg.get("hf_hub_id"), url=f"https://huggingface.co/{cfg.get('hf_hub_id')}",
                license=cfg.get("license") or (LICENSES[fam][0] if fam else "请核对 model card"),
                pretrain_data=LICENSES[fam][1] if fam else "请核对 model card",
                paper=cfg.get("paper_ids"))
    save_json(info, path + ".json")
    print(info)


def manifest(args):
    rows, total = [], 0
    files = sorted(glob.glob(os.path.join(args.model_dir, "*", "best.pth")))
    thr = os.path.join(args.model_dir, "thresholds.json")
    if os.path.exists(thr):
        files.append(thr)
    for f in files:
        s = os.path.getsize(f)
        total += s
        rows.append(dict(file=os.path.relpath(f, os.path.dirname(args.model_dir)), bytes=s, sha256=sha256(f), counted=True))
    for f in sorted(glob.glob(os.path.join(args.pretrain_dir, "*.pth")) + glob.glob(os.path.join(args.pretrain_dir, "*.safetensors"))):
        rows.append(dict(file=os.path.relpath(f, os.path.dirname(args.pretrain_dir)), bytes=os.path.getsize(f),
                         sha256=sha256(f), counted=False, note="预训练初始化权重，推理不加载"))
    out = dict(inference_total_mb=round(total / 2**20, 2), limit_mb=500, ok=total <= 500 * 2**20, files=rows)
    save_json(out, os.path.join(os.path.dirname(os.path.abspath(args.model_dir)), "file_manifest.json"))
    for r in rows:
        print(f"{r['file']:55s} {r['bytes']/2**20:8.2f} MB  {r['sha256'][:16]}...  {'计入' if r['counted'] else '不计入'}")
    print(f"推理加载文件合计 {out['inference_total_mb']} MB / 500 MB -> {'OK' if out['ok'] else '超限!'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("download")
    a.add_argument("--backbone", required=True)
    a.add_argument("--out", default="../pretrain_model")
    b = sp.add_parser("manifest")
    b.add_argument("--model_dir", default="../model")
    b.add_argument("--pretrain_dir", default="../pretrain_model")
    args = ap.parse_args()
    {"download": download, "manifest": manifest}[args.cmd](args)
    sys.exit(0)
