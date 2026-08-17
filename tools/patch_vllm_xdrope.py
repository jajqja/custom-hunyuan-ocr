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
import ast
import os
import shutil
import subprocess
import sys

MARK = "# >>> hyocr-xdrope-patch"

# Patch 2: hunyuan_vision.get_xdrope_input_positions đọc
# `hf_config.rope_scaling["xdrope_section"]`, nhưng transformers 5.x đã dời
# rope_scaling sang rope_parameters và để lại None -> TypeError. Đọc mềm hơn.
V_OLD = '        xd_num = len(hf_config.rope_scaling["xdrope_section"])\n'
V_NEW = (
    "        " + MARK + "\n"
    "        _rs = getattr(hf_config, \"rope_scaling\", None) or {}\n"
    "        _sec = (\n"
    "            _rs.get(\"xdrope_section\")\n"
    "            or getattr(hf_config, \"xdrope_section\", None)\n"
    "            or (getattr(hf_config, \"rope_parameters\", None) or {}).get(\n"
    "                \"mrope_section\"\n"
    "            )\n"
    "        )\n"
    "        xd_num = len(_sec)\n"
    "        # <<< hyocr-xdrope-patch\n"
)

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


def module_file(python, module):
    out = subprocess.run([python, "-c", f"import {module} as m; print(m.__file__)"],
                         capture_output=True, text=True)
    if out.returncode:
        raise SystemExit(f"không import được {module} bằng {python}:\n{out.stderr.strip()}")
    return out.stdout.strip()


def apply(path, anchor, addition, replace=False):
    """Chèn sau `anchor`, hoặc thay thế `anchor`. Idempotent, có sao lưu."""
    src = open(path, encoding="utf-8").read()
    if MARK in src:
        print(f"  {os.path.basename(path)}: đã vá rồi")
        return False
    if anchor not in src:
        raise SystemExit(f"không tìm thấy điểm neo trong {path}:\n  {anchor.strip()}\n"
                         "vLLM đã đổi cấu trúc — phải vá tay.")
    backup = path + ".hyocr.bak"
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
    out = src.replace(anchor, addition if replace else anchor + addition, 1)
    # Thụt lề sai làm file vẫn ghi được nhưng vỡ lúc import. Kiểm trước khi ghi.
    try:
        ast.parse(out)
    except SyntaxError as err:
        raise SystemExit(f"{path}: bản vá tạo ra cú pháp hỏng ở dòng {err.lineno} "
                         f"({err.msg}) — không ghi.")
    open(path, "w", encoding="utf-8").write(out)
    print(f"  {os.path.basename(path)}: đã vá (sao lưu {os.path.basename(backup)})")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter của môi trường có vllm")
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    rope = module_file(args.python, "vllm.model_executor.layers.rotary_embedding")
    vision = module_file(args.python, "vllm.model_executor.models.hunyuan_vision")

    if args.revert:
        for path in (rope, vision):
            backup = path + ".hyocr.bak"
            if os.path.exists(backup):
                shutil.copy2(backup, path)
                print(f"đã khôi phục {path}")
        return 0

    apply(rope, ANCHOR, PATCH)
    apply(vision, V_OLD, V_NEW, replace=True)

    check = subprocess.run(
        [args.python, "-c",
         "from vllm.model_executor.layers.rotary_embedding import get_rope; print('import OK')"],
        capture_output=True, text=True)
    print(check.stdout.strip() or check.stderr.strip()[-400:])
    return 0 if check.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
