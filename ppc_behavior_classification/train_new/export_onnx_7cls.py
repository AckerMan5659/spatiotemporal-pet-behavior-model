# -*- coding: utf-8 -*-
"""
Phase 4 — Streaming Single-Frame ONNX Export.

将 RepViT-GRU 拆分为:
    输入:  (input_img[1,3,224,224], in_state[1,hidden])
    输出:  (output_logits[1,7],     out_state[1,hidden])

这一步是边缘部署 (C++/C#/Web) 的最终交付物：
    - 摄像头每送 1 帧, 维护一个 hidden state, 即可做 O(1) 复杂度行为识别。
    - 支持 FP32 (蒸馏 ema) 与 QAT 权重两种导出。

Usage:
    python export_onnx_7cls.py --config config.yaml --source qat       # default
    python export_onnx_7cls.py --config config.yaml --source fp32
"""

import os
import sys
import argparse
import warnings
import yaml
import torch
import torch.nn as nn

warnings.filterwarnings("ignore", category=UserWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL_DIR = os.path.join(ROOT, "model")
for p in (HERE, ROOT, MODEL_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from recognizers.gru_model import RepViT_GRU                        # noqa: E402


# ----------------------------------------------------------------------
class RepViT_GRU_SingleFrame(nn.Module):
    """单帧 + GRUCell 状态接口, 与 train/swin_wrapper.RepViT_GRU_2D 等价。"""

    def __init__(self, base_model):
        super().__init__()
        self.backbone = base_model.backbone
        self.gap = base_model.gap
        self.fc = base_model.fc

        in_size = base_model.gru.input_size
        hidden = base_model.gru.hidden_size
        self.gru_cell = nn.GRUCell(input_size=in_size, hidden_size=hidden)
        print(f"💉 transplant GRU -> GRUCell (hidden={hidden})")

        self.gru_cell.weight_ih.data = base_model.gru.weight_ih_l0.data.clone()
        self.gru_cell.weight_hh.data = base_model.gru.weight_hh_l0.data.clone()
        self.gru_cell.bias_ih.data = base_model.gru.bias_ih_l0.data.clone()
        self.gru_cell.bias_hh.data = base_model.gru.bias_hh_l0.data.clone()

    def forward(self, input_img, in_state):
        feats = self.backbone(input_img)
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        x = self.gap(feats).flatten(1)
        h_out = self.gru_cell(x, in_state)
        logits = self.fc(h_out)
        return logits, h_out


def fuse_repvit(module):
    if hasattr(module, "switch_to_deploy"):
        try:
            module.switch_to_deploy()
        except Exception:
            pass
    for child in module.children():
        fuse_repvit(child)


# ----------------------------------------------------------------------
def _load_base(num_classes, hidden, ckpt_path, qat=False):
    base = RepViT_GRU(num_classes=num_classes, hidden_dim=hidden, pretrained=False)
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"📥 loading {ckpt_path}")
        sd = torch.load(ckpt_path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        sd = {k.replace("module.", ""): v for k, v in sd.items()}

        if qat:
            # QAT wrapper 把 backbone / gap / gru / fc 平铺保存; 重命名回 base
            remap = {}
            for k, v in sd.items():
                nk = k
                # strip leading wrapper prefixes like "backbone.", "gru.", ...
                # they already match RepViT_GRU 内部命名, 这里只清理 wrapper 前缀
                if k.startswith("backbone.") or k.startswith("gru.") \
                        or k.startswith("fc.") or k.startswith("gap."):
                    remap[nk] = v
            sd = remap if remap else sd

        miss, unexp = base.load_state_dict(sd, strict=False)
        print(f"   loaded (missing={len(miss)}, unexpected={len(unexp)})")
    else:
        print(f"⚠️ checkpoint missing: {ckpt_path}; using random init")
    return base


def export(cfg_path, source="qat"):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ds_cfg = cfg["dataset"]
    e_cfg = cfg["export"]
    hidden = cfg["distillation"]["student_hidden_dim"]
    num_classes = ds_cfg["num_classes"]
    img = ds_cfg["imgsz"]

    os.makedirs(e_cfg["output_dir"], exist_ok=True)

    if source == "qat":
        ckpt = e_cfg["qat_checkpoint"]
        out_name = e_cfg["streaming_onnx"]
        qat_flag = True
    else:
        ckpt = e_cfg["fp32_checkpoint"]
        out_name = e_cfg["fp32_streaming_onnx"]
        qat_flag = False

    out_path = os.path.join(e_cfg["output_dir"], out_name)
    print(f"🚀 [Phase 4] Streaming ONNX export ({source}) -> {out_path}")

    base = _load_base(num_classes, hidden, ckpt, qat=qat_flag)
    base.eval()

    print("✨ fusing RepViT (switch_to_deploy)")
    fuse_repvit(base)

    model_export = RepViT_GRU_SingleFrame(base).eval()

    dummy_img = torch.randn(1, 3, img, img)
    dummy_state = torch.zeros(1, hidden)

    print(f"📦 exporting opset={e_cfg.get('opset', 18)}")
    try:
        torch.onnx.export(
            model_export, (dummy_img, dummy_state), out_path,
            export_params=True,
            opset_version=e_cfg.get("opset", 18),
            do_constant_folding=True,
            input_names=["input_img", "in_state"],
            output_names=["output_logits", "out_state"],
            dynamic_axes={
                "input_img": {0: "batch"},
                "in_state": {0: "batch"},
                "output_logits": {0: "batch"},
                "out_state": {0: "batch"},
            },
        )
        print(f"✅ exported -> {out_path}")
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"   file size: {size_mb:.2f} MB")
    except Exception as e:
        print(f"❌ export failed: {e}")
        raise


# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default=os.path.join(HERE, "config.yaml"))
    parser.add_argument("--source", type=str, default="qat",
                        choices=["qat", "fp32"],
                        help="qat = use Phase-3 weight, fp32 = use Phase-2 weight")
    args = parser.parse_args()
    export(args.config, source=args.source)
