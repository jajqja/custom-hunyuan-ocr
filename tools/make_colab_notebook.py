#!/usr/bin/env python3
"""Sinh notebooks/hyocr_sft_colab.ipynb. Nội dung bám theo docs/COLAB.md."""
import json
import os

CELLS = [
    ("md", """# SFT HunyuanOCR-1.5 base — Colab

Chạy từ trên xuống. Runtime: **A100** (`Runtime → Change runtime type`).
Chi tiết và cách xử lý sự cố: [`docs/COLAB.md`](https://github.com/jajqja/custom-hunyuan-ocr/blob/main/docs/COLAB.md).

Trước khi chạy, sửa hai biến ở cell **Cấu hình**.

Toàn bộ phần train chạy trong một **venv riêng** ở `/content/venv`, không đụng
tới Python hệ thống của Colab. Lý do ở cell bước 5."""),

    ("md", "## 1. GPU và PACK_LEN"),
    ("code", r"""import re, subprocess

out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"],
                     capture_output=True, text=True).stdout
print(out)
GB = int(re.search(r"(\d+)\s*MiB", out).group(1)) / 1024
PACK_LEN = 16384 if GB >= 70 else 8192 if GB >= 35 else 4096
print(f"{GB:.0f} GB -> PACK_LEN={PACK_LEN}")
# Đọc từ nvidia-smi chứ không import torch: cell này chạy trước khi venv tồn tại.
# Colab cấp A100 cả bản 40GB lẫn 80GB tuỳ lần, nên đừng đặt cứng. Trên 80GB có
# thể thử 20480 sau khi xem headroom ở lần chạy đầu."""),

    ("md", """## 2. Cấu hình

Cả hai repo đều **private** — dữ liệu có CCCD, họ tên, ngày sinh, địa chỉ
thường trú và ảnh thẻ có chân dung, vân tay. Token phải có quyền read (và
write nếu đẩy model) trên org `newai-vn`. Đừng chuyển repo nào sang public."""),
    ("code", """DATASET_REPO = "newai-vn/vietnamese-doc-ocr"
OUTPUT_REPO  = "newai-vn/newai-ocr-v2"       # nơi đẩy model sau khi train

REPO_URL = "https://github.com/jajqja/custom-hunyuan-ocr.git"
MODEL_DIR, DATA_DIR, WORK = "/content/HunyuanOCR", "/content/dataset", "/content/hyocr"
DRIVE_DIR = "/content/drive/MyDrive/hyocr_sft"
VENV, RUN_NAME = "/content/venv", "colab_run\""""),

    ("md", """## 3. Gắn Drive

Session Colab chết là mất sạch `/content`. **Không** ghi checkpoint thẳng vào
Drive — một checkpoint model 1B kèm optimizer state cỡ 8–12 GB, ghi qua FUSE
sẽ làm training đứng hình. Bước 9 lưu vào `/content` rồi rsync sang Drive."""),
    ("code", """from google.colab import drive
drive.mount('/content/drive')
import os; os.makedirs(DRIVE_DIR, exist_ok=True)"""),

    ("md", "## 4. Clone fork"),
    ("code", """!git clone -q {REPO_URL} {WORK}
%cd {WORK}
!git log --oneline -1"""),

    ("md", """## 5. Tạo venv và cài dependency

Dựng venv sạch thay vì cài đè lên Python của Colab, vì hai lý do:

1. **flash-attn là bắt buộc và chỉ có wheel cho một số bản torch nhất định.**
   `train/trainer.py:3` import `flash_attn_varlen_func` ngay đầu module, còn
   `train_hunyuan.py` truyền cứng `attn_implementation="flash_attention_2"` —
   không có đường lùi sang eager. Wheel build riêng cho từng tổ hợp (CUDA major,
   torch minor, cxx11 ABI, Python, arch) và torch luôn ra trước wheel vài tuần,
   nên Colab thường chạy bản torch chưa có wheel. Build from source tốn 40–60
   phút mỗi session.
2. Hạ torch của Python hệ thống làm hỏng nửa số gói Colab cài sẵn và bắt phải
   restart kernel. Trong venv thì không ảnh hưởng gì.

`--plan` hỏi GitHub Releases API xem bản torch mới nhất nào còn wheel, rồi cài
đúng bộ đó vào venv. Không đoán tên file, nên không mục theo thời gian.

Venv được tạo bằng `uv` chứ không phải `python -m venv`: ảnh Colab thiếu
`ensurepip` (Debian tách ra gói `python3-venv`) nên stdlib venv thoát mã 1.
Cell chạy lại được nhiều lần — có venv rồi thì bỏ qua bước tạo, không cài lại
6 GB. Mọi lệnh cài in stdout/stderr khi lỗi thay vì chỉ ném
`CalledProcessError` rỗng."""),
    ("code", """import json, os, shutil, subprocess, sys

VENV = "/content/venv"
PY = f"{VENV}/bin/python"


def run(cmd, quiet=False):
    r = subprocess.run(cmd, text=True, capture_output=True)
    if r.returncode and not quiet:
        print("$", " ".join(cmd))
        print(r.stdout[-3000:], r.stderr[-3000:], sep="")
    return r.returncode


# Dùng uv chứ không phải `python -m venv`: Debian tách ensurepip ra gói
# python3-venv riêng mà ảnh Colab không cài, nên stdlib venv thoát mã 1 và để
# lại một thư mục hỏng dở. uv tự mang pip, và cài nhanh hơn hẳn khi phải kéo
# về ~6 GB torch.
assert run([sys.executable, "-m", "pip", "install", "-q", "-U", "uv"]) == 0

if not os.path.exists(PY):                    # chạy lại cell không cài lại từ đầu
    shutil.rmtree(VENV, ignore_errors=True)   # dọn venv hỏng dở của lần trước
    assert run([sys.executable, "-m", "uv", "venv", "--seed", "--clear", VENV]) == 0
print("venv:", PY)


def pipi(*args):
    assert run([sys.executable, "-m", "uv", "pip", "install", "-q",
                "--python", PY, *args]) == 0, f"cài thất bại: {args[0]}"


plan = json.loads(subprocess.run(
    [sys.executable, "tools/pick_flash_attn.py", "--plan"],
    capture_output=True, text=True, check=True).stdout)
print(json.dumps(plan, indent=2))

pipi(f"torch=={plan['torch']}", "torchvision", "--index-url", plan["index_url"])
pipi(plan["wheel"])
# `hunyuan_vl` chỉ có từ transformers 5.13.0 (5.11/5.12 chưa có). Đừng ghim
# @82a06db: bản đó thiếu HunYuanVLVisionTransformer nên train/trainer.py không
# import được — xem CUSTOM_SFT.md.
pipi("transformers>=5.13", "accelerate", "deepspeed", "safetensors", "datasets",
     "sentencepiece", "tokenizers", "pillow", "opencv-python-headless", "numpy",
     "einops", "tensorboard", "tqdm", "binpacking", "lmdb", "huggingface_hub>=0.34")
print("venv xong")"""),

    ("md", """Từ đây mọi lệnh gọi script đều truyền thẳng interpreter của venv
(`PYTHON={VENV}/bin/python`, `TORCHRUN={VENV}/bin/torchrun`) thay vì dựa vào
`PATH`. Cả hai script cũng tự kiểm tra interpreter trước khi chạy và báo lỗi
rõ ràng nếu thiếu gói, thay vì để traceback rơi ra giữa chừng."""),
    ("code", """!{VENV}/bin/python -c "import torch, flash_attn; \\
print('torch', torch.__version__, '| cuda', torch.version.cuda, \\
      '| flash_attn', flash_attn.__version__, '| gpu', torch.cuda.get_device_name(0))\""""),

    ("md", """## 6. Đăng nhập HF, tải model + dataset

`notebook_login()` chạy ở kernel hệ thống (cần widget) nhưng ghi token vào
`~/.cache/huggingface/token`, dùng chung với venv."""),
    ("code", """!pip install -q -U "huggingface_hub>=0.34"
from huggingface_hub import notebook_login
notebook_login()   # token quyền read; dataset đang private"""),
    ("code", """!{VENV}/bin/hf download tencent/HunyuanOCR --local-dir {MODEL_DIR} --exclude "v1.0/*"
!{VENV}/bin/hf download {DATASET_REPO} --repo-type dataset --local-dir {DATA_DIR}
!ls {DATA_DIR}"""),

    ("md", """Prompt lấy từ `configs/ocr_prompt.md` **của repo này**, không trích
từ README dataset. Prompt là cấu hình của lần train nên phải được version cùng
code, và phải khớp từng chữ với prompt lúc suy luận — model chỉ thấy đúng một
chuỗi prompt trong suốt quá trình train."""),
    ("code", """print(open(f"{WORK}/configs/ocr_prompt.md", encoding="utf-8").read())"""),

    ("md", """## 7. Convert sang JSONL phẳng

`--flat` xuất `{"image","question","answer"}` — schema mà `VLDataset` đọc trực
tiếp khi `is_packed=False`. Không có bước pack: packing chỉ chạy được trên bản
transformers prerelease đã ngừng dùng, và với 1.015 mẫu thì nó không đáng."""),
    ("code", """!{VENV}/bin/python tools/makedata_to_hyocr.py \\
    --root {DATA_DIR} \\
    --prompt-file {WORK}/configs/ocr_prompt.md \\
    --out-dir {WORK}/data/flat --flat
# cột "thiếu ảnh" phải là 0"""),
    ("code", """import json

for sp in ("train", "validation"):
    rows = [json.loads(l) for l in open(f"{WORK}/data/flat/{sp}.jsonl")]
    print(f"{sp:11} {len(rows):5} mẫu | nhãn rỗng "
          f"{sum(not r['answer'].strip() for r in rows)}")

!mkdir -p {DRIVE_DIR}/flat && cp {WORK}/data/flat/*.jsonl {DRIVE_DIR}/flat/"""),

    ("md", """## 8. Train

Không có validation trong lúc train: `Trainer` chỉ nhận **một** collator dùng cho
cả train và eval. Muốn eval loss thì thêm
`EVAL_DATA=$WORK/data/flat/validation.jsonl EVAL_STEPS=32`. Nhưng thứ quyết định
chọn checkpoint nào là CER sinh thật, đo sau bằng `eval_checkpoint.py`."""),
    ("code", """import subprocess
# Đồng bộ output sang Drive mỗi 10 phút để session chết không mất checkpoint.
# Đích là DRIVE_DIR/output/, KHÔNG phải DRIVE_DIR: `--delete` xoá mọi thứ ở
# đích mà nguồn không có, nên trỏ vào thư mục gốc sẽ xoá luôn DRIVE_DIR/flat/
# vừa sao lưu ở bước 7.
subprocess.Popen(f"mkdir -p {DRIVE_DIR}/output; "
                 f"while true; do rsync -a --delete {WORK}/output/ "
                 f"{DRIVE_DIR}/output/; sleep 600; done", shell=True)
print("rsync nền đã chạy ->", f"{DRIVE_DIR}/output/")"""),
    # Đường dẫn viết thẳng: line magic không nội suy biến như dòng `!`.
    ("code", """%load_ext tensorboard
%tensorboard --logdir /content/hyocr/output"""),
    ("code", """!PYTHON={VENV}/bin/python \\
 TORCHRUN={VENV}/bin/torchrun \\
 MODEL_PATH={MODEL_DIR} \\
 TRAIN_DATA={WORK}/data/flat/train.jsonl \\
 PACKING=0 \\
 GRAD_ACCUM=16 \\
 PACK_LEN={PACK_LEN} \\
 RUN_NAME={RUN_NAME} \\
 SAVE_STEPS=25 \\
     bash scripts/sft_base_1gpu.sh"""),

    ("md", """320 step (1.015 mẫu / 16 accum × 5 epoch). Không có validation trong lúc
train — `Trainer` chỉ nhận một collator dùng cho cả train và eval. Thêm
`EVAL_DATA=$WORK/data/flat/validation.jsonl EVAL_STEPS=32` nếu muốn eval loss,
nhưng thứ quyết định chọn checkpoint là CER sinh thật, đo sau bằng
`tools/eval_checkpoint.py`.

OOM thì thử theo thứ tự: `TUNE_VISION=False` (đóng băng vision tower, rẻ nhất) →
hạ `PACK_LEN` xuống 8192 (ở chế độ không pack nó là độ dài tối đa của **một
trang**, đổi được ngay, không phải xử lý lại dữ liệu) →
`DEEPSPEED=scripts/zero2.json`."""),

    ("md", """## 9. Mất session → chạy lại

Làm lại cell 1–8 (kể cả dựng venv — `/content` đã bị xoá sạch), rồi kéo
checkpoint từ Drive về **trước** khi chạy lại cell train với đúng `RUN_NAME` cũ.
`train_hunyuan.py:221` tự `resume_from_checkpoint=True` khi thấy `checkpoint-*`
trong output dir."""),
    ("code", """!mkdir -p {WORK}/output {WORK}/data/flat
!rsync -a {DRIVE_DIR}/output/ {WORK}/output/
!rsync -a {DRIVE_DIR}/flat/ {WORK}/data/flat/
!ls {WORK}/output/{RUN_NAME} && ls {WORK}/data/flat"""),

    ("md", "## 10. Đẩy model đã train lên HF"),
    ("code", """!{VENV}/bin/hf upload {OUTPUT_REPO} {WORK}/output/{RUN_NAME} . \\
    --repo-type model --private"""),
]


def main():
    cells = []
    for kind, src in CELLS:
        lines = src.split("\n")
        body = [x + "\n" for x in lines[:-1]] + [lines[-1]]
        if kind == "md":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": body})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "source": body,
                          "execution_count": None, "outputs": []})
    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "A100"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "notebooks", "hyocr_sft_colab.ipynb")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    print(f"{out}: {len(cells)} cell")


if __name__ == "__main__":
    main()
