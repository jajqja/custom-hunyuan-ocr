#!/usr/bin/env python3
"""Sinh notebooks/hyocr_sft_colab.ipynb. Nội dung bám theo docs/COLAB.md."""
import json
import os

CELLS = [
    ("md", """# SFT HunyuanOCR-1.5 base — Colab

Chạy từ trên xuống. Runtime: **A100** (`Runtime → Change runtime type`).
Chi tiết và cách xử lý sự cố: [`docs/COLAB.md`](https://github.com/jajqja/custom-hunyuan-ocr/blob/main/docs/COLAB.md).

Trước khi chạy, sửa hai biến ở cell **Cấu hình**."""),

    ("md", "## 1. GPU và PACK_LEN"),
    ("code", """import subprocess, torch

print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"],
                     capture_output=True, text=True).stdout)
GB = torch.cuda.get_device_properties(0).total_memory / 1024**3
PACK_LEN = 16384 if GB >= 70 else 8192 if GB >= 35 else 4096
print(f"{GB:.0f} GB -> PACK_LEN={PACK_LEN}")
# Colab cấp A100-40GB, không phải 80GB. PACK_LEN=8192 ~ 2 trang/pack, đúng như mong đợi."""),

    ("md", "## 2. Cấu hình — **sửa hai dòng dưới**"),
    ("code", """DATASET_REPO = "<user>/vietnamese-doc-ocr"   # repo dataset đã đẩy lên HF
OUTPUT_REPO  = "<user>/hunyuanocr-vi-sft"    # nơi đẩy model sau khi train (bước 9)

REPO_URL = "https://github.com/jajqja/custom-hunyuan-ocr.git"
MODEL_DIR, DATA_DIR, WORK = "/content/HunyuanOCR", "/content/dataset", "/content/hyocr"
DRIVE_DIR = "/content/drive/MyDrive/hyocr_sft"
RUN_NAME = "colab_run\""""),

    ("md", """## 3. Gắn Drive

Session Colab chết là mất sạch `/content`. **Không** ghi checkpoint thẳng vào
Drive — một checkpoint model 1B kèm optimizer state cỡ 8–12 GB, ghi qua FUSE
sẽ làm training đứng hình. Bước 7 lưu vào `/content` rồi rsync sang Drive."""),
    ("code", """from google.colab import drive
drive.mount('/content/drive')
import os; os.makedirs(DRIVE_DIR, exist_ok=True)"""),

    ("md", "## 4. Clone fork"),
    ("code", """!git clone -q $REPO_URL $WORK
%cd $WORK
!git log --oneline -1"""),

    ("md", "## 5. Dependency"),
    ("code", """!pip install -q -U "transformers>=4.57" accelerate deepspeed safetensors \\
                   binpacking lmdb tensorboard "huggingface_hub>=0.34\""""),

    ("md", """`flash-attn` bắt buộc: `train/trainer.py:3` import `flash_attn_varlen_func`
ngay đầu module và `train_hunyuan.py` truyền cứng `attn_implementation="flash_attention_2"`.
Build from source mất 40–60 phút, nên lấy wheel dựng sẵn khớp runtime."""),
    ("code", """import sys, torch, subprocess

ver = "2.8.3"
tv  = ".".join(torch.__version__.split("+")[0].split(".")[:2])
cu  = "cu" + torch.version.cuda.split(".")[0]
abi = "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE"
py  = f"cp{sys.version_info.major}{sys.version_info.minor}"
whl = (f"https://github.com/Dao-AILab/flash-attention/releases/download/v{ver}/"
       f"flash_attn-{ver}+{cu}torch{tv}cxx11abi{abi}-{py}-{py}-linux_x86_64.whl")
print(whl)
subprocess.run(["pip", "install", "-q", whl], check=True)

# 404 = tổ hợp torch/CUDA/Python này chưa có wheel. Chọn tay ở
# https://github.com/Dao-AILab/flash-attention/releases theo 4 mảnh in ra trên."""),
    ("code", """from flash_attn.flash_attn_interface import flash_attn_varlen_func
print("flash-attn OK")"""),

    ("md", "## 6. Đăng nhập HF, tải model + dataset"),
    ("code", """from huggingface_hub import notebook_login
notebook_login()   # token quyền read; dataset đang private"""),
    ("code", """!hf download tencent/HunyuanOCR --local-dir $MODEL_DIR --exclude "v1.0/*"
!hf download $DATASET_REPO --repo-type dataset --local-dir $DATA_DIR
!ls $DATA_DIR"""),

    ("md", """Prompt không nằm thành file riêng trong repo dataset — nó ở trong
`README.md`, trong khối ```` ```text ````."""),
    ("code", """import re, pathlib

readme = pathlib.Path(f"{DATA_DIR}/README.md").read_text(encoding="utf-8")
prompt = re.search(r"````text\\n(.*?)\\n````", readme, re.S).group(1).strip()
pathlib.Path("/content/ocr_prompt.md").write_text(prompt, encoding="utf-8")
print(prompt)"""),

    ("md", "## 7. Convert sang raw JSONL"),
    ("code", """!python tools/makedata_to_hyocr.py \\
    --root $DATA_DIR \\
    --prompt-file /content/ocr_prompt.md \\
    --out-dir $WORK/data/raw \\
    --data-list $WORK/data/data_list.txt
# cột "thiếu ảnh" phải là 0"""),

    ("md", """## 8. Pack

`NUM_PROCESSES=2` chứ không phải 32: Colab chỉ có ~12 vCPU và mỗi process load
một processor riêng. Mất 10–20 phút vì phải mở từng ảnh để đếm vision token."""),
    ("code", """!MODEL_PATH=$MODEL_DIR \\
 INPUT_LIST=$WORK/data/data_list.txt \\
 PACK_OUTPUT=$WORK/data/packed/train_$PACK_LEN.jsonl \\
 PACK_LEN=$PACK_LEN \\
 NUM_PROCESSES=2 THREADS_PER_PROCESS=4 \\
 FOREGROUND=1 \\
     bash scripts/pack_data.sh"""),
    ("code", """import json

n = t = 0
for line in open(f"{WORK}/data/packed/train_{PACK_LEN}.jsonl"):
    p = json.loads(line); n += 1; t += sum(x["num_tokens"] for x in p)
print(f"{n} pack, trung bình {t/n:.0f} token/pack")

!mkdir -p $DRIVE_DIR/packed && cp $WORK/data/packed/*.jsonl $DRIVE_DIR/packed/"""),

    ("md", "## 9. Train"),
    ("code", """import subprocess
# Đồng bộ output sang Drive mỗi 10 phút để session chết không mất checkpoint.
subprocess.Popen(f"while true; do rsync -a --delete {WORK}/output/ {DRIVE_DIR}/; "
                 f"sleep 600; done", shell=True)
print("rsync nền đã chạy")"""),
    ("code", """%load_ext tensorboard
%tensorboard --logdir $WORK/output"""),
    ("code", """!MODEL_PATH=$MODEL_DIR \\
 TRAIN_DATA=$WORK/data/packed/train_$PACK_LEN.jsonl \\
 PACK_LEN=$PACK_LEN \\
 RUN_NAME=$RUN_NAME \\
 SAVE_STEPS=25 \\
     bash scripts/sft_base_1gpu.sh"""),

    ("md", """OOM trên 40GB thì thử theo thứ tự: `TUNE_VISION=False` (đóng băng vision
tower, rẻ nhất) → `PACK_LEN=4096` (phải pack lại) → `DEEPSPEED=scripts/zero2.json`."""),

    ("md", """## 10. Mất session → chạy lại

Làm lại cell 1–7, rồi kéo checkpoint từ Drive về **trước** khi chạy lại cell
train với đúng `RUN_NAME` cũ. `train_hunyuan.py:221` tự
`resume_from_checkpoint=True` khi thấy `checkpoint-*` trong output dir."""),
    ("code", """!mkdir -p $WORK/output $WORK/data/packed
!rsync -a $DRIVE_DIR/ $WORK/output/
!cp $DRIVE_DIR/packed/*.jsonl $WORK/data/packed/ 2>/dev/null
!ls $WORK/output/$RUN_NAME"""),

    ("md", "## 11. Đẩy model đã train lên HF"),
    ("code", """!hf upload $OUTPUT_REPO $WORK/output/$RUN_NAME . --repo-type model --private"""),
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
