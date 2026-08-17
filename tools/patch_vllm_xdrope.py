#!/usr/bin/env python3
"""
Vá vLLM để HunyuanOCR đi đúng nhánh XDRotaryEmbedding.

Config gốc của HunyuanOCR khai báo:

    "rope_scaling": {"type": "xdrope", "xdrope_section": [16,16,16,16], "alpha": 1000.0, ...}

transformers 5.x chuẩn hoá nó khi load thành:

    {"rope_type": "dynamic", "mrope_section": [16,16,16,16], "alpha": 1000.0, ...}

vLLM đọc bản đã chuẩn hoá đó, nên `get_rope()` rơi vào nhánh

    elif scaling_type == "dynamic":
        if "alpha" in rope_parameters:  -> DynamicNTKAlphaRotaryEmbedding   (RoPE 1 chiều)

trong khi model cần

    elif scaling_type == "xdrope":      -> XDRotaryEmbedding(xdrope_section=...)

Hệ quả: rope 1 chiều nhận position 4 trục, engine chết ở profile_run với
`view(..., 3*s18, -1, 128)` hoặc, ở chế độ eager, `rotary_embedding: query, key
and positions must have the same batch_size and seq_len`.

Nhánh xdrope vẫn còn nguyên trong vLLM — chỉ là không còn config nào đi vào
được. Bản vá này khôi phục đường đi đó, không đụng tới toán.

    python tools/patch_vllm_xdrope.py --python /content/vllmenv/bin/python
    python tools/patch_vllm_xdrope.py --python ... --revert
"""
import argparse
import os
import shutil
import subprocess
import sys

MARK = "# >>> hyocr-xdrope-patch"

PATCH = f'''{MARK}
    # transformers 5.x đổi tên {{"type":"xdrope","xdrope_section":[...]}} thành
    # {{"rope_type":"dynamic","mrope_section":[...]}}, khiến get_rope() dựng RoPE
    # 1 chiều cho một model dùng position 4 trục. Trả lại nhánh xdrope.
    if (
        scaling_type == "dynamic"
        and "alpha" in rope_parameters
        and "mrope_section" in rope_parameters
        and "xdrope_section" not in rope_parameters
    ):
        rope_parameters = dict(rope_parameters)
        rope_parameters["xdrope_section"] = rope_parameters["mrope_section"]
        scaling_type = "xdrope"
# <<< hyocr-xdrope-patch
'''

ANCHOR = '    scaling_type = rope_parameters.get("rope_type", "default")\n'


def target_file(python):
    out = subprocess.run(
        [python, "-c",
         "import vllm.model_executor.layers.rotary_embedding as m; print(m.__file__)"],
        capture_output=True, text=True)
    if out.returncode:
        raise SystemExit(f"không import được vllm bằng {python}:\n{out.stderr.strip()}")
    return out.stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter của môi trường có vllm")
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    path = target_file(args.python)
    backup = path + ".hyocr.bak"
    src = open(path, encoding="utf-8").read()

    if args.revert:
        if not os.path.exists(backup):
            raise SystemExit(f"không có bản sao lưu {backup}")
        shutil.copy2(backup, path)
        print(f"đã khôi phục {path}")
        return 0

    if MARK in src:
        print(f"{path}: đã vá rồi, bỏ qua")
        return 0
    if ANCHOR not in src:
        raise SystemExit(
            f"không tìm thấy điểm neo trong {path}.\n"
            f"Cần dòng: {ANCHOR.strip()}\n"
            "vLLM đã đổi cấu trúc get_rope() — phải vá tay.")

    if not os.path.exists(backup):
        shutil.copy2(path, backup)
    open(path, "w", encoding="utf-8").write(src.replace(ANCHOR, ANCHOR + PATCH, 1))
    print(f"đã vá {path}\nsao lưu   {backup}")

    check = subprocess.run(
        [args.python, "-c",
         "from vllm.model_executor.layers.rotary_embedding import get_rope; print('import OK')"],
        capture_output=True, text=True)
    print(check.stdout.strip() or check.stderr.strip()[-400:])
    return 0 if check.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
