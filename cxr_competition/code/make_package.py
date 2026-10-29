# -*- coding: utf-8 -*-
"""打包提交材料：队伍名_最终提交材料.zip
默认不放入预训练权重二进制（公开可下载，推理不加载），改为放入：
  pretrain_model/*.json（来源链接 + 许可证 + SHA-256）+ download_pretrain.py（下载并校验哈希）
用法:
  python make_package.py --team 队伍名
  python make_package.py --team 队伍名 --include_pretrain   # 空间允许时连权重一起放
"""
import os
import glob
import json
import shutil
import argparse
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DOWNLOAD_SCRIPT = r'''# -*- coding: utf-8 -*-
"""下载本方案使用的 ImageNet 预训练权重，并校验 SHA-256 与申报值一致。
用法: python download_pretrain.py [--mirror https://hf-mirror.com]"""
import os, sys, json, glob, hashlib, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
base = "https://huggingface.co"
if "--mirror" in sys.argv:
    base = sys.argv[sys.argv.index("--mirror") + 1].rstrip("/")

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

ok = True
for meta in sorted(glob.glob(os.path.join(HERE, "*.json"))):
    info = json.load(open(meta, encoding="utf-8"))
    dst = os.path.join(HERE, info["file"])
    if not os.path.exists(dst):
        url = f"{base}/{info['hf_hub_id']}/resolve/main/model.safetensors"
        print("下载", url)
        urllib.request.urlretrieve(url, dst)
    got = sha256(dst)
    same = got == info["sha256"]
    ok &= same
    print(f"{info['file']}: {'SHA-256 一致' if same else '不一致! ' + got}")
sys.exit(0 if ok else 1)
'''


def add_tree(zf, src, arc, pattern="**/*", exclude=()):
    for p in sorted(glob.glob(os.path.join(src, pattern), recursive=True)):
        if os.path.isfile(p) and not any(e in p for e in exclude):
            zf.write(p, os.path.join(arc, os.path.relpath(p, src)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    ap.add_argument("--submission", default=os.path.join(ROOT, "submission.csv"), help="历史最高分的 csv")
    ap.add_argument("--include_pretrain", action="store_true")
    ap.add_argument("--limit_mb", type=float, default=500)
    args = ap.parse_args()

    pre = os.path.join(ROOT, "pretrain_model")
    with open(os.path.join(pre, "download_pretrain.py"), "w", encoding="utf-8") as f:
        f.write(DOWNLOAD_SCRIPT)
    metas = sorted(glob.glob(os.path.join(pre, "*.json")))
    if not metas:
        raise SystemExit("pretrain_model/ 下没有 *.json，请先运行 run_all.sh 第0步或 tools.py register")

    name = f"{args.team}_最终提交材料"
    out = os.path.join(ROOT, name + ".zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        add_tree(zf, os.path.join(ROOT, "code"), f"{name}/code", "*.py")
        add_tree(zf, os.path.join(ROOT, "code"), f"{name}/code", "requirements.txt")
        for p in sorted(glob.glob(os.path.join(ROOT, "model", "*", "best.pth"))):
            zf.write(p, f"{name}/model/{os.path.relpath(p, os.path.join(ROOT, 'model'))}")
        thr = os.path.join(ROOT, "model", "thresholds.json")
        if os.path.exists(thr):
            zf.write(thr, f"{name}/model/thresholds.json")
        for p in metas + [os.path.join(pre, "download_pretrain.py")]:
            zf.write(p, f"{name}/pretrain_model/{os.path.basename(p)}")
        if args.include_pretrain:
            for m in metas:
                w = os.path.join(pre, json.load(open(m, encoding="utf-8"))["file"])
                zf.write(w, f"{name}/pretrain_model/{os.path.basename(w)}")
        add_tree(zf, os.path.join(ROOT, "licenses"), f"{name}/licenses")
        for f in ["cost.py", "README.md", "file_manifest.json"]:
            if os.path.exists(os.path.join(ROOT, f)):
                zf.write(os.path.join(ROOT, f), f"{name}/{f}")
        zf.write(args.submission, f"{name}/submission.csv")
        sub_dir = os.path.dirname(os.path.abspath(args.submission))
        for f in ["test_meta.csv", "image_preds.csv"]:   # cost.py 需要：逐图元数据 + 逐图概率（一致性核验）
            if os.path.exists(os.path.join(sub_dir, f)):
                zf.write(os.path.join(sub_dir, f), f"{name}/{f}")

    with zipfile.ZipFile(out) as zf:
        unz = sum(i.file_size for i in zf.infolist())
    zmb, umb = os.path.getsize(out) / 2**20, unz / 2**20
    print(f"写出 {out}\n  压缩后 {zmb:.1f} MB，解压后 {umb:.1f} MB，上限 {args.limit_mb} MB -> "
          f"{'OK' if max(zmb, umb) <= args.limit_mb else '超限!'}")
    print("  预训练权重: " + ("已包含" if args.include_pretrain else "未包含（附来源链接 + SHA-256 + download_pretrain.py）"))


if __name__ == "__main__":
    main()
