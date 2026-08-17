#!/usr/bin/env python3
"""
Ghép một `checkpoint-*` thành thư mục model tự đứng được, sẵn sàng đẩy lên Hub.

Một checkpoint của Trainer **không load lại được một mình**: nó có trọng số +
config + tokenizer, nhưng thiếu image processor (`train_hunyuan.py` gọi
`processor.save_pretrained` đúng một lần, vào thư mục gốc, sau khi train xong).
Nó cũng kèm `optimizer.pt` / `scheduler.pt` / `rng_state.pth` — cỡ vài GB, chỉ
để resume, không ai cần khi inference.

    python tools/export_checkpoint.py \\
        --checkpoint ./output/colab_run/checkpoint-300 \\
        --base ./HunyuanOCR \\
        --out ./export/hyocr-vi-ckpt300
"""
import argparse
import json
import os
import shutil
import sys

# Chỉ để resume, không đưa lên Hub.
SKIP = {"optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json",
        "training_args.bin", "scaler.pt"}
# Lấy từ model gốc nếu checkpoint không có.
FROM_BASE = ["preprocessor_config.json", "processor_config.json",
             "video_preprocessor_config.json", "chat_template.jinja",
             "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
             "generation_config.json"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--base", required=True, help="thư mục HunyuanOCR gốc")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    copied, skipped = [], []
    for name in sorted(os.listdir(args.checkpoint)):
        src = os.path.join(args.checkpoint, name)
        if name in SKIP or name.startswith("rng_state") or os.path.isdir(src):
            skipped.append(name)
            continue
        shutil.copy2(src, os.path.join(args.out, name))
        copied.append(name)

    filled = []
    for name in FROM_BASE:
        dst = os.path.join(args.out, name)
        src = os.path.join(args.base, name)
        if not os.path.exists(dst) and os.path.exists(src):
            shutil.copy2(src, dst)
            filled.append(name)

    print(f"từ checkpoint : {', '.join(copied)}")
    print(f"bỏ qua        : {', '.join(skipped) or '—'}")
    print(f"lấy từ base   : {', '.join(filled) or '—'}")

    missing = [n for n in ("config.json", "preprocessor_config.json")
               if not os.path.exists(os.path.join(args.out, n))]
    weights = [n for n in os.listdir(args.out) if n.endswith((".safetensors", ".bin"))]
    if missing or not weights:
        print(f"\n[LỖI] thiếu {missing or 'trọng số model'} — thư mục này chưa load được",
              file=sys.stderr)
        return 1

    size = sum(os.path.getsize(os.path.join(args.out, f)) for f in os.listdir(args.out))
    print(f"\n{args.out}: {len(os.listdir(args.out))} file, {size / 1e9:.2f} GB")
    try:
        with open(os.path.join(args.out, "config.json"), encoding="utf-8") as fh:
            print("architectures:", json.load(fh).get("architectures"))
    except json.JSONDecodeError:
        print("[cảnh báo] config.json không parse được")
    return 0


if __name__ == "__main__":
    sys.exit(main())
