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

    ("md", "## 2. Cấu hình — **sửa hai dòng dưới**"),
    ("code", """DATASET_REPO = "<user>/vietnamese-doc-ocr"   # repo dataset đã đẩy lên HF
OUTPUT_REPO  = "<user>/hunyuanocr-vi-sft"    # nơi đẩy model sau khi train

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
    ("code", """!git clone -q $REPO_URL $WORK
%cd $WORK
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

Nếu `python -m venv` hỏng (Colab hay thiếu `ensurepip` vì Debian tách nó ra gói
`python3-venv`), cell tự chuyển sang `uv venv --seed`. Mọi lệnh cài đều in
stdout/stderr khi lỗi thay vì chỉ ném `CalledProcessError`."""),
    ("code", """import json, os, subprocess, sys

VENV = "/content/venv"
PY = f"{VENV}/bin/python"


def run(cmd, quiet=False):
    r = subprocess.run(cmd, text=True, capture_output=True)
    if r.returncode and not quiet:
        print("$", " ".join(cmd))
        print(r.stdout[-3000:], r.stderr[-3000:], sep="")
    return r.returncode


# Tạo venv. `python -m venv` trên Colab hay hỏng vì Debian tách ensurepip ra gói
# python3-venv riêng; uv tự mang pip nên không phụ thuộc vào đó.
UV = False
if run([sys.executable, "-m", "venv", VENV], quiet=True) != 0:
    print("stdlib venv không dùng được (thường thiếu ensurepip) -> chuyển sang uv")
    assert run([sys.executable, "-m", "pip", "install", "-q", "uv"]) == 0
    assert run([sys.executable, "-m", "uv", "venv", "--seed", VENV]) == 0
    UV = True
assert os.path.exists(PY), "không tạo được venv"
print("venv:", PY, "| uv:", UV)


def pipi(*args):
    cmd = ([sys.executable, "-m", "uv", "pip", "install", "-q", "--python", PY, *args]
           if UV else [PY, "-m", "pip", "install", "-q", *args])
    assert run(cmd) == 0, f"cài thất bại: {args[0]}"


plan = json.loads(subprocess.run(
    [sys.executable, "tools/pick_flash_attn.py", "--plan"],
    capture_output=True, text=True, check=True).stdout)
print(json.dumps(plan, indent=2))

if not UV:
    pipi("-U", "pip", "setuptools", "wheel")
pipi(f"torch=={plan['torch']}", "torchvision", "--index-url", plan["index_url"])
pipi(plan["wheel"])
pipi("transformers>=4.57", "accelerate", "deepspeed", "safetensors", "datasets",
     "sentencepiece", "tokenizers", "pillow", "opencv-python-headless", "numpy",
     "einops", "tensorboard", "tqdm", "binpacking", "lmdb", "huggingface_hub>=0.34")
print("venv xong")"""),

    ("md", """Từ đây mọi lệnh shell đều chạy với `PATH=$VENV/bin:$PATH`, để `python`
và `torchrun` trong các script `scripts/*.sh` trỏ vào venv."""),
    ("code", """!$VENV/bin/python -c "import torch, flash_attn; \\
print('torch', torch.__version__, '| cuda', torch.version.cuda, \\
      '| flash_attn', flash_attn.__version__, '| gpu', torch.cuda.get_device_name(0))\""""),

    ("md", """## 6. Đăng nhập HF, tải model + dataset

`notebook_login()` chạy ở kernel hệ thống (cần widget) nhưng ghi token vào
`~/.cache/huggingface/token`, dùng chung với venv."""),
    ("code", """!pip install -q -U "huggingface_hub>=0.34"
from huggingface_hub import notebook_login
notebook_login()   # token quyền read; dataset đang private"""),
    ("code", """!$VENV/bin/hf download tencent/HunyuanOCR --local-dir $MODEL_DIR --exclude "v1.0/*"
!$VENV/bin/hf download $DATASET_REPO --repo-type dataset --local-dir $DATA_DIR
!ls $DATA_DIR"""),

    ("md", """Prompt không nằm thành file riêng trong repo dataset — nó ở trong
`README.md`, trong khối ```` ```text ````."""),
    ("code", r"""import re, pathlib

readme = pathlib.Path(f"{DATA_DIR}/README.md").read_text(encoding="utf-8")
prompt = re.search(r"````text\n(.*?)\n````", readme, re.S).group(1).strip()
pathlib.Path("/content/ocr_prompt.md").write_text(prompt, encoding="utf-8")
print(prompt)"""),

    ("md", "## 7. Convert sang raw JSONL"),
    ("code", """!$VENV/bin/python tools/makedata_to_hyocr.py \\
    --root $DATA_DIR \\
    --prompt-file /content/ocr_prompt.md \\
    --out-dir $WORK/data/raw \\
    --data-list $WORK/data/data_list.txt
# cột "thiếu ảnh" phải là 0"""),

    ("md", """## 8. Pack

`NUM_PROCESSES=2` chứ không phải 32: Colab chỉ có ~12 vCPU và mỗi process load
một processor riêng. Mất 10–20 phút vì phải mở từng ảnh để đếm vision token."""),
    ("code", """!PATH=$VENV/bin:$PATH \\
 MODEL_PATH=$MODEL_DIR \\
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
    ("code", """!PATH=$VENV/bin:$PATH \\
 MODEL_PATH=$MODEL_DIR \\
 TRAIN_DATA=$WORK/data/packed/train_$PACK_LEN.jsonl \\
 PACK_LEN=$PACK_LEN \\
 RUN_NAME=$RUN_NAME \\
 SAVE_STEPS=25 \\
     bash scripts/sft_base_1gpu.sh"""),

    ("md", """OOM thì thử theo thứ tự: `TUNE_VISION=False` (đóng băng vision tower, rẻ
nhất) → hạ `PACK_LEN` một bậc (phải pack lại) → `DEEPSPEED=scripts/zero2.json`."""),

    ("md", """## 10. Mất session → chạy lại

Làm lại cell 1–8 (kể cả dựng venv — `/content` đã bị xoá sạch), rồi kéo
checkpoint từ Drive về **trước** khi chạy lại cell train với đúng `RUN_NAME` cũ.
`train_hunyuan.py:221` tự `resume_from_checkpoint=True` khi thấy `checkpoint-*`
trong output dir."""),
    ("code", """!mkdir -p $WORK/output $WORK/data/packed
!rsync -a $DRIVE_DIR/ $WORK/output/
!cp $DRIVE_DIR/packed/*.jsonl $WORK/data/packed/ 2>/dev/null
!ls $WORK/output/$RUN_NAME"""),

    ("md", "## 11. Đẩy model đã train lên HF"),
    ("code", """!$VENV/bin/hf upload $OUTPUT_REPO $WORK/output/$RUN_NAME . \\
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
